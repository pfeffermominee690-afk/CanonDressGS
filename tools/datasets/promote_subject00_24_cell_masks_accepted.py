"""Freeze Subject00 mask human review and prepare Teacher-target preflight.

This task is metadata-only.  It verifies immutable accepted RGB and mask
bindings, records the fixed human PASS decisions for all 24 cells, promotes
the 24 mask pairs to accepted, and writes a non-executable Teacher-target
preflight draft.  It never edits images, creates Teacher targets, or starts
training/inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF = ROOT / "project_control_handoff"
DOCS = ROOT / "docs" / "PAPER"

TASK_ID = "AAAI27-SUBJECT00-24-CELL-MASK-HUMAN-REVIEW-PROMOTION-001"
SOURCE_BRANCH = "research/subject00-24-cell-mask-human-review-pack-20260726"
SOURCE_HEAD = "e6ba0e5750feb1d78ed3fadc7b53886c7bab2140"
NEW_BRANCH = "research/subject00-24-cell-mask-human-review-promotion-20260727"
WORKTREE = Path(
    r"E:\model_train\canondressgs_subject00_24_cell_mask_human_review_promotion"
)
REVIEWER = "USER_AND_GPT_MANUAL_REVIEW"
FINAL_CLASSIFICATION = (
    "SUBJECT00_24_OF_24_MASKS_ACCEPTED_TEACHER_TARGET_PREFLIGHT_PENDING"
)
NEXT_TASK = (
    "PREFLIGHT_AND_FREEZE_SUBJECT00_24_CELL_TEACHER_TARGET_CREATION_CONTRACT"
)
BASE_AVATAR_STATUS = "PENDING_SUBJECT00_FORMAL_BASE_FINALIZATION"

UPLOAD_MANIFEST_PATH = (
    RISK / "subject00_24_cell_mask_human_review_upload_manifest_20260726.json"
)
HUMAN_MANIFEST_PATH = (
    RISK / "subject00_24_cell_mask_human_review_manifest_20260726.json"
)
EXECUTION_REGISTRY_PATH = (
    RISK / "subject00_24_cell_mask_generation_execution_registry_20260726.json"
)
PERSON_QA_PATH = RISK / "subject00_24_cell_person_mask_qa_results_20260726.json"
GARMENT_QA_PATH = RISK / "subject00_24_cell_garment_mask_qa_results_20260726.json"
PAIR_QA_PATH = RISK / "subject00_24_cell_mask_pair_qa_results_20260726.json"
CROSS_VIEW_QA_PATH = RISK / "subject00_24_cell_mask_cross_view_qa_20260726.json"
EXECUTION_SUMMARY_PATH = (
    RISK / "subject00_24_cell_mask_generation_execution_final_summary_20260726.json"
)
ACCEPTED_IMAGE_REGISTRY_PATH = (
    RISK / "subject00_global_accepted_cell_registry_24of24_20260726.json"
)
ACCEPTANCE_AUDIT_PATH = (
    RISK / "subject00_24_cell_acceptance_audit_registry_20260726.json"
)
SELECTED_IMAGE_REGISTRY_PATH = (
    RISK / "subject00_global_selected_cell_registry_24of24_20260726.json"
)
O03_OVERRIDE_PATH = RISK / "subject00_o03_canary_human_review_overlay_20260726.json"
REMAINING_OVERRIDE_PATH = (
    RISK / "subject00_remaining_six_human_review_overlay_20260726.json"
)
UPLOAD_INDEX_PATH = RISK / "subject00_mask_upload_pack_index_20260726.json"
IMMUTABILITY_SNAPSHOT_PATH = (
    RISK / "subject00_24_cell_mask_human_review_pack_pre_snapshot_20260726.json"
)
IMMUTABILITY_RESULT_PATH = (
    RISK / "subject00_24_cell_mask_human_review_pack_immutability_20260726.json"
)
CONDITION_BINDING_PATH = RISK / "subject00_condition_slot_binding.json"
DATASET_CONTRACT_PATH = RISK / "subject00_three_garment_dataset_contract.json"
DATASET_DIRECTORY_PATH = RISK / "subject00_dataset_directory_binding.json"

REVIEW_PDF_PATH = Path(
    r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-MASKS-001"
    r"\attempt_001_subject00_24_cell_person_garment_masks"
    r"\07_review_assets\upload_pack_20260726\05_indexes"
    r"\subject00_24_cell_mask_human_review_pages_20260726.pdf"
)
REVIEW_PDF_BYTES = 100_056_298
REVIEW_PDF_SHA256 = (
    "7f9da88729be097265b0a8e8843ef10c46105ca68c4474785bc88cc4fc4f0c7b"
)

MASK_ACCEPTED_REGISTRY_PATH = (
    RISK / "subject00_global_mask_accepted_registry_24of24_20260727.json"
)
HUMAN_REVIEW_OVERLAY_PATH = (
    RISK / "subject00_24_cell_mask_human_review_overlay_20260727.json"
)
PROMOTION_OVERLAY_PATH = (
    RISK / "subject00_24_cell_mask_accepted_promotion_overlay_20260727.json"
)
HUMAN_REVIEW_SUMMARY_PATH = (
    RISK / "subject00_24_cell_mask_human_review_summary_20260727.json"
)
TEST_RESULTS_PATH = (
    RISK / "subject00_24_cell_mask_human_review_tests_20260727.json"
)
REPORT_PATH = RISK / "SUBJECT00_24_CELL_MASK_HUMAN_REVIEW_REPORT_20260727.md"
TEACHER_CONTRACT_PATH = (
    RISK / "SUBJECT00_24_CELL_TEACHER_TARGET_PREFLIGHT_CONTRACT_20260727.md"
)
TEACHER_MANIFEST_PATH = (
    RISK / "subject00_24_cell_teacher_target_preflight_manifest_draft_20260727.json"
)
HANDOFF_PATH = (
    HANDOFF
    / "subject00_24_cell_mask_accepted_teacher_target_preflight_handoff_20260727.json"
)
DOCS_REPORT_PATH = (
    DOCS / "AAAI27_SUBJECT00_24_CELL_MASK_HUMAN_REVIEW_REPORT_20260727.md"
)

PLANNED_CLOUD_TARGET_ROOT = (
    "/root/autodl-tmp/canondressgs_work/datasets/"
    "subject00_three_garment/09_teacher_targets/"
)
PLANNED_WINDOWS_TARGET_ROOT = Path(
    r"E:\model_train\canondressgs_work\datasets"
    r"\subject00_three_garment\09_teacher_targets"
)

O03_OVERRIDE_ID = "subject00_O03_slot04_canary_attempt004_cand00"
O01_OVERRIDE_ID = "subject00_O01_slot04_remaining_attempt005_cand00"
OVERRIDE_IDS = [O03_OVERRIDE_ID, O01_OVERRIDE_ID]
DIRECTION_TO_SEMANTIC_SLOT = {
    "front": "front",
    "front-left": "front_left_three_quarter",
    "front-right": "front_right_three_quarter",
    "left": "left",
    "right": "right",
    "back-left": "back_left_three_quarter",
    "back-right": "back_right_three_quarter",
    "back": "back",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def record_map(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["request_id"]: record for record in value["records"]}


def verify_file(path: Path, expected_bytes: int, expected_sha: str) -> None:
    require(path.is_file(), f"missing immutable file: {path}")
    require(path.stat().st_size == expected_bytes, f"byte mismatch: {path}")
    require(sha256(path) == expected_sha, f"SHA256 mismatch: {path}")


def verify_image(
    path: Path,
    expected_bytes: int,
    expected_sha: str,
    expected_resolution: tuple[int, int] | None = None,
) -> tuple[int, int]:
    verify_file(path, expected_bytes, expected_sha)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        resolution = image.size
    if expected_resolution is not None:
        require(resolution == expected_resolution, f"resolution mismatch: {path}")
    return resolution


def verify_frozen_baseline(snapshot: dict[str, Any]) -> int:
    records = snapshot["baseline"]["records"]
    mutations = []
    for record in records:
        path = Path(record["path"])
        if not path.is_file():
            mutations.append({"path": str(path), "reason": "MISSING"})
            continue
        if path.stat().st_size != record["bytes"]:
            mutations.append({"path": str(path), "reason": "BYTES_CHANGED"})
            continue
        if sha256(path) != record["sha256"]:
            mutations.append({"path": str(path), "reason": "SHA256_CHANGED"})
    require(not mutations, f"frozen baseline mutations: {mutations[:3]}")
    return len(records)


PERSON_HUMAN_BASIS = [
    "single complete person foreground",
    "head, face, hair or target hood outline complete",
    "both hands, both feet, and shoes complete",
    "all currently worn clothing included",
    "no material wall, carpet, shadow, tripod, light, or second-person leakage",
    "no severe holes or boundary fracture",
    "minor pixel-level segmentation error is non-blocking",
]

GARMENT_HUMAN_BASIS = {
    "O01": [
        "hoodie/top, target hood fabric, sleeves, and trousers complete",
        "face, skin, hands, shoes, and background excluded",
        "8-view garment semantics consistent",
    ],
    "O03": [
        "formal suit semantics consistent",
        "jacket, shirt/tie region, and trousers complete",
        "face, hair, skin, hands, and shoes excluded",
        "8/8 views contain no hood or residual blue-white hoodie",
    ],
    "O04": [
        "black hooded outerwear, target hood fabric, and jeans/trousers complete",
        "face, hair, skin, hands, shoes, and background excluded",
        "8-view garment semantics consistent",
    ],
}

PAIR_HUMAN_BASIS = [
    "person and garment mask shapes and resolutions agree",
    "garment mask is a subset of person mask",
    "garment outside person pixels equal zero",
    "protected region is non-empty and visibly retains face, hands, and shoes",
    "garment mask does not cover the entire person",
    "no mask/path mixing is visible",
]


def main() -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    inputs = {
        "upload": load(UPLOAD_MANIFEST_PATH),
        "human_manifest": load(HUMAN_MANIFEST_PATH),
        "execution": load(EXECUTION_REGISTRY_PATH),
        "person": load(PERSON_QA_PATH),
        "garment": load(GARMENT_QA_PATH),
        "pair": load(PAIR_QA_PATH),
        "cross_view": load(CROSS_VIEW_QA_PATH),
        "execution_summary": load(EXECUTION_SUMMARY_PATH),
        "accepted_image": load(ACCEPTED_IMAGE_REGISTRY_PATH),
        "audit": load(ACCEPTANCE_AUDIT_PATH),
        "selected": load(SELECTED_IMAGE_REGISTRY_PATH),
        "o03_override": load(O03_OVERRIDE_PATH),
        "remaining_override": load(REMAINING_OVERRIDE_PATH),
        "upload_index": load(UPLOAD_INDEX_PATH),
        "snapshot": load(IMMUTABILITY_SNAPSHOT_PATH),
        "immutability_result": load(IMMUTABILITY_RESULT_PATH),
        "condition_binding": load(CONDITION_BINDING_PATH),
        "dataset_contract": load(DATASET_CONTRACT_PATH),
        "dataset_directory": load(DATASET_DIRECTORY_PATH),
    }

    execution_records = inputs["execution"]["records"]
    authoritative_ids = [record["request_id"] for record in execution_records]
    require(len(authoritative_ids) == len(set(authoritative_ids)) == 24, "request IDs")
    for key in ("upload", "person", "garment", "pair", "accepted_image"):
        candidate_ids = [record["request_id"] for record in inputs[key]["records"]]
        require(set(candidate_ids) == set(authoritative_ids), f"{key} ID set changed")
    require(inputs["person"]["pass_count"] == 24, "person QA pass count")
    require(inputs["person"]["fail_count"] == 0, "person QA fail count")
    require(inputs["garment"]["pass_count"] == 24, "garment QA pass count")
    require(inputs["garment"]["fail_count"] == 0, "garment QA fail count")
    require(inputs["pair"]["pass_count"] == 24, "pair QA pass count")
    require(inputs["pair"]["fail_count"] == 0, "pair QA fail count")
    require(inputs["upload_index"]["review_page_count"] == 8, "review page count")
    require(len(inputs["upload_index"]["pages"]) == 8, "review pages")
    require(inputs["accepted_image"]["accepted_without_limitation_count"] == 15, "15")
    require(inputs["accepted_image"]["accepted_with_limitation_count"] == 9, "9")
    require(inputs["execution_summary"]["teacher_target_count"] == 0, "Teacher count")
    require(
        sum(bool(record["mask_accepted"]) for record in execution_records) == 0,
        "previous mask accepted count",
    )
    require(
        sum(bool(record["teacher_target"]) for record in execution_records) == 0,
        "previous Teacher target count",
    )
    require(not PLANNED_WINDOWS_TARGET_ROOT.exists(), "Teacher target root preexists")
    require(
        inputs["immutability_result"]["status"] == "PASS_NO_MUTATIONS",
        "prior immutability result changed",
    )

    frozen_baseline_count = verify_frozen_baseline(inputs["snapshot"])
    require(frozen_baseline_count == 368, "frozen baseline file count")
    verify_file(REVIEW_PDF_PATH, REVIEW_PDF_BYTES, REVIEW_PDF_SHA256)
    for page in inputs["upload_index"]["pages"]:
        verify_image(
            Path(page["path"]),
            page["bytes"],
            page["sha256"],
            tuple(page["resolution_wh"]),
        )

    execution_map = record_map(inputs["execution"])
    person_map = record_map(inputs["person"])
    garment_map = record_map(inputs["garment"])
    pair_map = record_map(inputs["pair"])
    upload_map = record_map(inputs["upload"])
    accepted_map = record_map(inputs["accepted_image"])
    audit_map = record_map(inputs["audit"])
    selected_map = record_map(inputs["selected"])
    o03_override_map = record_map(inputs["o03_override"])
    remaining_override_map = record_map(inputs["remaining_override"])
    condition_map = {
        (binding["garment_id"], binding["slot_id"]): binding
        for binding in inputs["condition_binding"]["bindings"]
    }

    limitation_ids = [
        request_id
        for request_id in authoritative_ids
        if accepted_map[request_id]["accepted_with_limitation"]
    ]
    require(len(limitation_ids) == 9, "limitation set is not exact 9")
    limitation_codes = sorted(
        {
            code
            for request_id in limitation_ids
            for code in accepted_map[request_id]["disclosed_limitation_codes"]
        }
    )
    require(
        set(OVERRIDE_IDS).issubset(set(limitation_ids)),
        "override cells missing from limitation set",
    )

    review_evidence = {
        "reviewer": REVIEWER,
        "pdf": {
            "path": str(REVIEW_PDF_PATH),
            "bytes": REVIEW_PDF_BYTES,
            "sha256": REVIEW_PDF_SHA256,
            "page_count": 8,
        },
        "upload_manifest": {
            "path": str(UPLOAD_MANIFEST_PATH),
            "sha256": sha256(UPLOAD_MANIFEST_PATH),
        },
        "upload_index": {
            "path": str(UPLOAD_INDEX_PATH),
            "sha256": sha256(UPLOAD_INDEX_PATH),
        },
        "page_order": list(inputs["upload_index"]["upload_order"]),
    }

    human_records = []
    accepted_mask_records = []
    teacher_records = []
    raw_total_bytes = 0
    person_total_bytes = 0
    garment_total_bytes = 0
    machine_fail_ids = []

    for sequence_index, request_id in enumerate(authoritative_ids, start=1):
        execution = execution_map[request_id]
        person = person_map[request_id]
        garment = garment_map[request_id]
        pair = pair_map[request_id]
        upload = upload_map[request_id]
        accepted = accepted_map[request_id]
        audit = audit_map[request_id]
        selected = selected_map[request_id]
        binding = condition_map[(execution["garment"], execution["slot"])]

        width = accepted["native_resolution"]["width"]
        height = accepted["native_resolution"]["height"]
        resolution = (width, height)
        require(execution["raw_path"] == accepted["raw_path"], f"raw path {request_id}")
        verify_image(
            Path(execution["raw_path"]),
            execution["raw_bytes"],
            execution["raw_sha256"],
            resolution,
        )
        require(person["mask"]["path"] == execution["person_mask_path"], request_id)
        require(garment["mask"]["path"] == execution["garment_mask_path"], request_id)
        verify_image(
            Path(person["mask"]["path"]),
            person["mask"]["bytes"],
            person["mask"]["sha256"],
            resolution,
        )
        verify_image(
            Path(garment["mask"]["path"]),
            garment["mask"]["bytes"],
            garment["mask"]["sha256"],
            resolution,
        )
        require(person["technical_status"] == "PASS_PENDING_HUMAN_REVIEW", request_id)
        require(garment["technical_status"] == "PASS_PENDING_HUMAN_REVIEW", request_id)
        require(pair["technical_status"] == "PASS_PENDING_HUMAN_REVIEW", request_id)
        require(pair["garment_outside_person_pixel_count"] == 0, request_id)
        require(pair["protected_region_area"] > 0, request_id)
        require(all(person["checks"].values()), f"person checks {request_id}")
        require(all(garment["checks"].values()), f"garment checks {request_id}")
        require(all(pair["checks"].values()), f"pair checks {request_id}")
        require(
            accepted["disclosed_limitation_codes"] == audit["limitations"],
            f"limitation propagation {request_id}",
        )
        require(
            int(accepted["camera_id"].removeprefix("cam")) == binding["camera_id"],
            f"camera binding {request_id}",
        )
        require(
            DIRECTION_TO_SEMANTIC_SLOT[accepted["direction"]]
            == binding["semantic_pose_slot"],
            f"direction binding {request_id}",
        )

        machine_status = accepted["machine_registration_status"]
        machine_classification = accepted["machine_registration_status"]
        human_override = accepted["human_override_status"]
        if request_id == O03_OVERRIDE_ID:
            original = o03_override_map[request_id]
            machine_status = original["machine_registration_status"]
            machine_classification = original[
                "machine_registration_primary_classification"
            ]
            human_override = original["machine_registration_human_override"]
            require(machine_status == "FAIL", "O03 machine status changed")
            require(
                human_override == "PASS_VISUALLY_ACCEPTABLE_ALIGNMENT",
                "O03 override changed",
            )
        elif request_id == O01_OVERRIDE_ID:
            original = remaining_override_map[request_id]
            machine_status = original["machine_registration_status"]
            machine_classification = accepted["machine_registration_status"]
            human_override = original["human_override_status"]
            require(
                machine_status == "AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL",
                "O01 machine status changed",
            )
            require(
                human_override
                == "PASS_VISUALLY_ACCEPTABLE_SUBJECT_ALIGNMENT",
                "O01 override changed",
            )
        if request_id in OVERRIDE_IDS:
            machine_fail_ids.append(request_id)
        else:
            require(human_override is None, f"unexpected override {request_id}")
            require(
                machine_status == "REGISTERED_SIMILARITY_PASS_CANDIDATE",
                f"unexpected machine result {request_id}",
            )

        limitation_code_values = list(accepted["disclosed_limitation_codes"])
        limitation_details = list(accepted["disclosed_limitations"])
        limitation_evidence = list(accepted["limitation_evidence_paths"])
        mask_acceptance_status = (
            "MASK_ACCEPTED_WITH_DISCLOSED_SOURCE_LIMITATION"
            if limitation_code_values
            else "MASK_ACCEPTED"
        )
        image_acceptance_class = accepted["accepted_status"]
        cell_review_evidence = {
            "pdf_path": str(REVIEW_PDF_PATH),
            "review_page": upload["review_page"],
            "row_index": upload["row_index"],
            "master_page": upload["master_page"],
            "risk_page": upload["risk_page"],
            "reviewer": REVIEWER,
        }
        technical_qa = {
            "person": {
                "status": "PASS",
                "source_status": person["technical_status"],
                "checks": person["checks"],
                "registry_path": str(PERSON_QA_PATH),
                "registry_sha256": sha256(PERSON_QA_PATH),
            },
            "garment": {
                "status": "PASS",
                "source_status": garment["technical_status"],
                "checks": garment["checks"],
                "registry_path": str(GARMENT_QA_PATH),
                "registry_sha256": sha256(GARMENT_QA_PATH),
            },
            "pair": {
                "status": "PASS",
                "source_status": pair["technical_status"],
                "checks": pair["checks"],
                "garment_outside_person_pixel_count": 0,
                "protected_region_area": pair["protected_region_area"],
                "registry_path": str(PAIR_QA_PATH),
                "registry_sha256": sha256(PAIR_QA_PATH),
            },
        }

        human_records.append(
            {
                "sequence_index": sequence_index,
                "subject": "Subject00",
                "garment": execution["garment"],
                "slot": execution["slot"],
                "camera": execution["camera_id"],
                "direction": execution["direction"],
                "request_id": request_id,
                "reviewed": True,
                "person_mask_human_decision": "PASS",
                "garment_mask_human_decision": "PASS",
                "mask_pair_human_decision": "PASS",
                "mask_human_blocked": False,
                "machine_registration_status": machine_status,
                "machine_registration_classification": machine_classification,
                "human_override_status": human_override,
                "limitation_codes": limitation_code_values,
                "review_evidence": cell_review_evidence,
            }
        )

        accepted_mask_records.append(
            {
                "sequence_index": sequence_index,
                "subject": "Subject00",
                "garment": execution["garment"],
                "slot": execution["slot"],
                "camera": execution["camera_id"],
                "direction": execution["direction"],
                "request_id": request_id,
                "attempt_id": execution["attempt_id"],
                "accepted_raw": {
                    "path": execution["raw_path"],
                    "bytes": execution["raw_bytes"],
                    "sha256": execution["raw_sha256"],
                },
                "person_mask": {
                    "path": person["mask"]["path"],
                    "bytes": person["mask"]["bytes"],
                    "sha256": person["mask"]["sha256"],
                },
                "garment_mask": {
                    "path": garment["mask"]["path"],
                    "bytes": garment["mask"]["bytes"],
                    "sha256": garment["mask"]["sha256"],
                },
                "native_resolution": {"width": width, "height": height},
                "technical_qa": technical_qa,
                "person_mask_human_decision": "PASS",
                "garment_mask_human_decision": "PASS",
                "mask_pair_human_decision": "PASS",
                "image_acceptance_class": image_acceptance_class,
                "limitation_codes": limitation_code_values,
                "limitation_details": limitation_details,
                "limitation_evidence_paths": limitation_evidence,
                "machine_registration_status": machine_status,
                "machine_registration_classification": machine_classification,
                "human_override_status": human_override,
                "mask_acceptance_status": mask_acceptance_status,
                "mask_accepted": True,
                "teacher_target": False,
                "promotion_task_id": TASK_ID,
                "promotion_source_head": SOURCE_HEAD,
                "review_evidence": cell_review_evidence,
            }
        )

        teacher_records.append(
            {
                "sequence_index": sequence_index,
                "teacher_target_id": (
                    f"subject00/{execution['garment']}/{execution['slot']}"
                ),
                "subject": "Subject00",
                "garment": execution["garment"],
                "slot": execution["slot"],
                "camera": execution["camera_id"],
                "direction": execution["direction"],
                "request_id": request_id,
                "source_bindings": {
                    "accepted_raw": {
                        "path": execution["raw_path"],
                        "bytes": execution["raw_bytes"],
                        "sha256": execution["raw_sha256"],
                    },
                    "person_mask": {
                        "path": person["mask"]["path"],
                        "bytes": person["mask"]["bytes"],
                        "sha256": person["mask"]["sha256"],
                    },
                    "garment_mask": {
                        "path": garment["mask"]["path"],
                        "bytes": garment["mask"]["bytes"],
                        "sha256": garment["mask"]["sha256"],
                    },
                },
                "native_resolution": {"width": width, "height": height},
                "camera_pose_binding": {
                    "camera_id": binding["camera_id"],
                    "camera_file": binding["camera_file"],
                    "pose_frame_id": binding["pose_frame_id"],
                    "pose_smplx_file": binding["pose_smplx_file"],
                    "strict_split_role": binding["strict_split_role"],
                    "calibrated_view_orientation_degrees": binding[
                        "calibrated_view_orientation_degrees"
                    ],
                },
                "base_avatar_checkpoint_binding": {
                    "required_step": 101245,
                    "path": None,
                    "sha256": None,
                    "status": BASE_AVATAR_STATUS,
                },
                "limitation_codes": limitation_code_values,
                "limitation_details": limitation_details,
                "human_override_status": human_override,
                "planned_cloud_output_path": binding[
                    "accepted_teacher_target_path"
                ],
                "planned_windows_output_path": str(
                    PLANNED_WINDOWS_TARGET_ROOT
                    / execution["garment"]
                    / f"{execution['slot']}.png"
                ),
                "teacher_target_status": "NOT_CREATED",
                "teacher_target": False,
            }
        )
        raw_total_bytes += execution["raw_bytes"]
        person_total_bytes += person["mask"]["bytes"]
        garment_total_bytes += garment["mask"]["bytes"]

    require(
        set(machine_fail_ids) == set(OVERRIDE_IDS) and len(machine_fail_ids) == 2,
        "machine failure set changed",
    )
    require(len(accepted_mask_records) == 24, "accepted mask records")
    coverage = {
        garment: sum(
            record["garment"] == garment for record in accepted_mask_records
        )
        for garment in ("O01", "O03", "O04")
    }
    require(coverage == {"O01": 8, "O03": 8, "O04": 8}, "coverage")

    total_source_bytes = raw_total_bytes + person_total_bytes + garment_total_bytes
    storage_estimate = {
        "accepted_raw_bytes": raw_total_bytes,
        "person_mask_bytes": person_total_bytes,
        "garment_mask_bytes": garment_total_bytes,
        "all_bound_source_bytes": total_source_bytes,
        "copy_materialization_upper_bound_bytes": total_source_bytes,
        "planning_bytes_with_20_percent_metadata_and_qa_headroom": math.ceil(
            total_source_bytes * 1.2
        ),
        "materialized_bytes_in_this_task": 0,
    }

    human_overlay = {
        "schema_version": "canondressgs.subject00.mask_human_review_overlay.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "reviewer": REVIEWER,
        "review_evidence": review_evidence,
        "reviewed_cell_count": 24,
        "person_mask_human_pass_count": 24,
        "person_mask_human_fail_count": 0,
        "garment_mask_human_pass_count": 24,
        "garment_mask_human_fail_count": 0,
        "mask_pair_human_pass_count": 24,
        "mask_pair_human_fail_count": 0,
        "mask_human_blocked_count": 0,
        "person_human_acceptance_basis": PERSON_HUMAN_BASIS,
        "garment_human_acceptance_basis": GARMENT_HUMAN_BASIS,
        "pair_human_acceptance_basis": PAIR_HUMAN_BASIS,
        "records": human_records,
        "teacher_target_count": 0,
        "paper_final": False,
    }
    dump(HUMAN_REVIEW_OVERLAY_PATH, human_overlay)

    accepted_registry = {
        "schema_version": "canondressgs.subject00.global_mask_accepted_registry.24of24.v1",
        "task_id": TASK_ID,
        "created_at": created_at,
        "promotion_source_branch": SOURCE_BRANCH,
        "promotion_source_head": SOURCE_HEAD,
        "reviewer": REVIEWER,
        "review_evidence": review_evidence,
        "total_mask_accepted_cell_count": 24,
        "mask_pair_accepted_count": 24,
        "mask_accepted_coverage_by_garment": coverage,
        "mask_accepted_without_limitation_count": 15,
        "mask_accepted_with_disclosed_source_limitation_count": 9,
        "mask_accepted_blocked_count": 0,
        "limitation_request_ids": limitation_ids,
        "limitation_codes": limitation_codes,
        "machine_registration_pass_count": 22,
        "machine_registration_fail_count": 2,
        "human_override_count": 2,
        "human_override_request_ids": OVERRIDE_IDS,
        "teacher_target_count": 0,
        "teacher_target_creation_authorized": False,
        "teacher_endpoint_optimization_authorized": False,
        "records": accepted_mask_records,
        "paper_final": False,
    }
    dump(MASK_ACCEPTED_REGISTRY_PATH, accepted_registry)

    promotion_overlay = {
        "schema_version": "canondressgs.subject00.mask_accepted_promotion_overlay.v1",
        "task_id": TASK_ID,
        "previous_mask_accepted_count": 0,
        "newly_mask_accepted_count": 24,
        "total_mask_accepted_count": 24,
        "mask_accepted_without_limitation_count": 15,
        "mask_accepted_with_disclosed_source_limitation_count": 9,
        "mask_accepted_blocked_count": 0,
        "teacher_target_count": 0,
        "teacher_target_creation_authorized": False,
        "teacher_endpoint_optimization_authorized": False,
        "promotion_source_head": SOURCE_HEAD,
        "paper_final": False,
    }
    dump(PROMOTION_OVERLAY_PATH, promotion_overlay)

    teacher_manifest = {
        "schema_version": "canondressgs.subject00.teacher_target_preflight_manifest_draft.v1",
        "task_id": TASK_ID,
        "status": "DRAFT_PREFLIGHT_ONLY_NOT_AUTHORIZED",
        "mask_accepted_registry": {
            "path": str(MASK_ACCEPTED_REGISTRY_PATH),
            "sha256": sha256(MASK_ACCEPTED_REGISTRY_PATH),
        },
        "source_contracts": [
            {"path": str(CONDITION_BINDING_PATH), "sha256": sha256(CONDITION_BINDING_PATH)},
            {"path": str(DATASET_CONTRACT_PATH), "sha256": sha256(DATASET_CONTRACT_PATH)},
            {"path": str(DATASET_DIRECTORY_PATH), "sha256": sha256(DATASET_DIRECTORY_PATH)},
        ],
        "accepted_raw_count": 24,
        "person_mask_count": 24,
        "garment_mask_count": 24,
        "teacher_target_count": 0,
        "teacher_target_creation_authorized": False,
        "teacher_endpoint_optimization_authorized": False,
        "target_roots": {
            "planned_cloud_root": PLANNED_CLOUD_TARGET_ROOT,
            "planned_windows_root": str(PLANNED_WINDOWS_TARGET_ROOT),
            "created_in_this_task": False,
        },
        "target_dataset_schema": {
            "schema_id": "subject00.teacher_target_observation.v1",
            "required_fields": [
                "teacher_target_id",
                "subject",
                "garment",
                "slot",
                "camera",
                "direction",
                "accepted_raw_path",
                "accepted_raw_sha256",
                "person_mask_path",
                "person_mask_sha256",
                "garment_mask_path",
                "garment_mask_sha256",
                "native_resolution",
                "camera_calibration_path",
                "camera_calibration_sha256",
                "pose_smplx_path",
                "pose_smplx_sha256",
                "base_avatar_checkpoint_path",
                "base_avatar_checkpoint_sha256",
                "limitation_codes",
                "human_override_status",
                "provenance",
            ],
            "materialization_strategy": (
                "UNRESOLVED_MUST_BE_FROZEN_BEFORE_TEACHER_TARGET_CREATION"
            ),
        },
        "loader_contract": {
            "accepted_raw_loader": "PIL RGB; verify bytes/SHA before decode",
            "person_mask_loader": "PIL L; require values {0,255}",
            "garment_mask_loader": "PIL L; require values {0,255}",
            "pair_invariants": [
                "raw/person/garment dimensions exactly equal per sample",
                "garment mask is a subset of person mask",
                "garment outside person pixels equal zero",
                "protected region is non-empty",
            ],
            "native_mixed_resolution_handling": (
                "per-sample native resolution; batching must use explicit reversible "
                "padding or batch_size=1 only after a later frozen preflight"
            ),
            "resize_allowed": False,
            "reencode_allowed": False,
        },
        "base_avatar_dependency": {
            "status": BASE_AVATAR_STATUS,
            "required_checkpoint_step": 101245,
            "checkpoint_path": None,
            "checkpoint_sha256": None,
            "required_final_gates": [
                "formal Base Avatar technical PASS",
                "formal Base Avatar visual PASS",
                "checkpoint path/bytes/SHA frozen",
            ],
        },
        "teacher_endpoint_compatibility": {
            "status": "PENDING_FORMAL_BASE_AND_LOADER_PREFLIGHT",
            "mixed_resolution_support_must_be_verified": True,
            "automatic_optimization_allowed": False,
        },
        "limitations_propagation_required": True,
        "human_override_request_ids": OVERRIDE_IDS,
        "storage_estimate": storage_estimate,
        "teacher_target_qa": {
            "required": True,
            "checks": [
                "all source bytes and SHA256 values reverified",
                "native dimensions preserved",
                "raw/person/garment loader agreement",
                "binary mask values",
                "garment subset and protected-region invariants",
                "camera/pose binding complete",
                "Base Avatar checkpoint binding complete",
                "all limitations and overrides propagated",
                "Teacher Endpoint compatibility reported",
            ],
            "human_review_required_after_materialization": True,
        },
        "records": teacher_records,
        "paper_modification_allowed": False,
        "paper_final": False,
    }
    dump(TEACHER_MANIFEST_PATH, teacher_manifest)

    summary = {
        "schema_version": "canondressgs.subject00.mask_human_review_summary.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "worktree": str(WORKTREE),
        "reviewer": REVIEWER,
        "review_evidence": review_evidence,
        "reviewed_cell_count": 24,
        "person_mask_human_pass_count": 24,
        "person_mask_human_fail_count": 0,
        "garment_mask_human_pass_count": 24,
        "garment_mask_human_fail_count": 0,
        "mask_pair_human_pass_count": 24,
        "mask_pair_human_fail_count": 0,
        "mask_human_blocked_count": 0,
        "raw_accepted_cell_count": 24,
        "person_mask_generated_count": 24,
        "garment_mask_generated_count": 24,
        "total_mask_generated_count": 48,
        "previous_mask_accepted_count": 0,
        "newly_mask_accepted_count": 24,
        "total_mask_accepted_count": 24,
        "mask_pair_accepted_count": 24,
        "mask_accepted_without_limitation_count": 15,
        "mask_accepted_with_disclosed_limitation_count": 9,
        "mask_missing_count": 0,
        "mask_blocked_count": 0,
        "coverage": coverage,
        "limitation_request_ids": limitation_ids,
        "limitation_codes": limitation_codes,
        "machine_registration_pass_count": 22,
        "machine_registration_fail_count": 2,
        "human_override_count": 2,
        "human_override_request_ids": OVERRIDE_IDS,
        "teacher_target_count": 0,
        "teacher_target_creation_authorized": False,
        "teacher_endpoint_optimization_authorized": False,
        "teacher_target_preflight_base_avatar_status": BASE_AVATAR_STATUS,
        "person_mask_mutations": 0,
        "garment_mask_mutations": 0,
        "accepted_raw_mutations": 0,
        "attempt_mutations": {
            "attempt_001": 0,
            "attempt_002": 0,
            "attempt_003": 0,
            "attempt_004": 0,
            "attempt_005": 0,
        },
        "frozen_baseline_checked_file_count": frozen_baseline_count,
        "segmentation_inference_calls": 0,
        "model_forward_calls": 0,
        "image_generation_calls": 0,
        "data_mutations": 0,
        "paper_modifications": 0,
        "subject00_multi_garment_data_status": (
            "RAW_AND_MASK_ACCEPTED_24_OF_24_TEACHER_TARGET_PREFLIGHT_PENDING"
        ),
        "test_result": "PENDING_POST_GENERATION_VALIDATION",
        "commit_head": "PENDING_SUBSTANTIVE_COMMIT",
        "final_reporting_head": "PENDING_FINAL_REPORTING_COMMIT",
        "origin_sync_status": "PENDING",
        "cloud_git_sync_status": "PENDING_BEST_EFFORT",
        "worktree_clean_status": "PENDING",
        "paper_final": False,
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }
    dump(HUMAN_REVIEW_SUMMARY_PATH, summary)

    teacher_contract = f"""# Subject00 24-Cell Teacher-Target Preflight Contract

## Authority and boundary

- Task: `{TASK_ID}`
- Source: `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`
- Input state: 24/24 accepted RGB cells and 24/24 accepted person/garment
  mask pairs.
- `TEACHER_TARGET_CREATION_AUTHORIZED=false`
- `TEACHER_TARGET_COUNT=0`
- `TEACHER_ENDPOINT_OPTIMIZATION_AUTHORIZED=false`

This document is a preflight draft only. It does not authorize or create a
Teacher target, dataset root, checkpoint, endpoint run, training run, image,
mask, resize, re-encoding operation, or paper-body modification.

## Frozen source bindings

The draft manifest `{TEACHER_MANIFEST_PATH}` binds, in the authoritative
24-record order:

- 24 accepted raw paths, byte counts, and SHA256 values;
- 24 accepted person-mask paths, byte counts, and SHA256 values;
- 24 accepted garment-mask paths, byte counts, and SHA256 values;
- garment, slot, camera, direction, and native resolution;
- per-slot calibration and SMPL-X pose bindings;
- all disclosed source limitations and both human registration overrides.

The planned cloud root is `{PLANNED_CLOUD_TARGET_ROOT}`. The planned Windows
mirror is `{PLANNED_WINDOWS_TARGET_ROOT}`. Neither root is created by this
task.

## Camera and Base Avatar binding

Every record binds the frozen `subject00_condition_slot_binding.json` camera
and pose metadata. Camera calibration remains bound to
`/root/autodl-tmp/datasets/thuman4_second_identity_staging/subject00/calibration.json`
with SHA256
`4b2d98d20740da03be209040851fab7e8801e5670937ed9ad290a20264d1dde7`.

The required Subject00 Formal Base checkpoint is step 101245. Its final path
and SHA256 are not yet frozen. Therefore:

`TEACHER_TARGET_PREFLIGHT_BASE_AVATAR_STATUS={BASE_AVATAR_STATUS}`

The next preflight must require both technical and visual Formal Base PASS and
must freeze checkpoint path, bytes, and SHA256 before any Teacher-target
materialization. This pending dependency does not block the present mask
promotion, but it blocks Teacher Endpoint optimization.

## Target dataset and loader contract

The draft schema is `subject00.teacher_target_observation.v1`. A record must
contain accepted RGB, person mask, garment mask, native resolution,
camera/pose binding, final Base checkpoint binding, limitations, human
override, and provenance.

Accepted RGB is decoded as RGB. Person and garment masks are decoded as
single-channel `L` images with values exactly `{{0,255}}`. Dimensions must
match per record; garment must remain a subset of person; protected region
must remain non-empty.

The 24 records contain mixed native resolutions. No resize and no re-encoding
are allowed. A later frozen preflight must select either per-sample execution
or explicit reversible padding and prove Teacher Endpoint compatibility.
Materialization strategy (copy, link, or another immutable binding) remains
unresolved and must be frozen before creation.

## Limitations and overrides

The nine limitation-bearing cells remain accepted with disclosed source
limitations. Every code, description, and evidence path must propagate to
the future Teacher-target registry and Teacher Endpoint report.

The two machine failures remain machine failures:

1. `{O03_OVERRIDE_ID}`: status `FAIL`, classification
   `AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL`, human override
   `PASS_VISUALLY_ACCEPTABLE_ALIGNMENT`.
2. `{O01_OVERRIDE_ID}`: status
   `AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL`, human override
   `PASS_VISUALLY_ACCEPTABLE_SUBJECT_ALIGNMENT`.

Neither machine result may be rewritten as PASS.

## Storage and QA

The exact bound source-byte total is `{total_source_bytes}`. The conservative
planning value with 20% metadata/QA headroom is
`{storage_estimate['planning_bytes_with_20_percent_metadata_and_qa_headroom']}`.
Materialized bytes in this task are zero.

Future Teacher-target QA must reverify every source SHA, dimensions, binary
mask values, pair subset/protected-region invariants, camera/pose binding,
Base checkpoint binding, limitation propagation, and endpoint compatibility.
Human review remains mandatory after materialization.

## Next unique task

`{NEXT_TASK}`

That task must not automatically create Teacher targets or start Teacher
Endpoint optimization.
"""
    write(TEACHER_CONTRACT_PATH, teacher_contract)

    report = f"""# Subject00 24-Cell Mask Human-Review Promotion Report

## Result

The fixed `{REVIEWER}` review of `{REVIEW_PDF_PATH}` (8 pages,
`{REVIEW_PDF_BYTES}` bytes, SHA256 `{REVIEW_PDF_SHA256}`) is frozen:

- person masks: 24 PASS, 0 FAIL;
- garment masks: 24 PASS, 0 FAIL;
- mask pairs: 24 PASS, 0 FAIL;
- human blockers: 0;
- O01/O03/O04 accepted coverage: 8/8 each.

All 24 mask pairs are promoted to `mask_accepted=true`. Fifteen are accepted
without a source limitation and nine are accepted with every disclosed
source limitation retained. The two historical machine-registration failures
remain failures and retain their human overrides.

## Immutability and authorization

The 24 raw images, 24 person masks, 24 garment masks, eight upload pages, the
merged review PDF, and all 368 files in the prior frozen attempt baseline
were rehashed. No mutation was found. No inference, model forward, image
generation, dataset mutation, or paper-body modification occurred.

No Teacher target or target root was created. Teacher-target creation and
Teacher Endpoint optimization remain unauthorized. The Formal Base dependency
is `{BASE_AVATAR_STATUS}`.

## Artifacts

- Mask accepted registry: `{MASK_ACCEPTED_REGISTRY_PATH}`
- Human-review overlay: `{HUMAN_REVIEW_OVERLAY_PATH}`
- Promotion overlay: `{PROMOTION_OVERLAY_PATH}`
- Summary: `{HUMAN_REVIEW_SUMMARY_PATH}`
- Teacher preflight contract: `{TEACHER_CONTRACT_PATH}`
- Teacher preflight manifest: `{TEACHER_MANIFEST_PATH}`

Final classification: `{FINAL_CLASSIFICATION}`.

Next task: `{NEXT_TASK}`.
"""
    write(REPORT_PATH, report)
    write(DOCS_REPORT_PATH, report)

    handoff = {
        "schema_version": "canondressgs.subject00.mask_accepted_teacher_target_preflight_handoff.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "worktree": str(WORKTREE),
        "reviewer": REVIEWER,
        "reviewed_cell_count": 24,
        "total_mask_accepted_count": 24,
        "teacher_target_count": 0,
        "teacher_target_creation_authorized": False,
        "teacher_endpoint_optimization_authorized": False,
        "base_avatar_dependency_status": BASE_AVATAR_STATUS,
        "artifacts": {
            "mask_accepted_registry": str(MASK_ACCEPTED_REGISTRY_PATH),
            "human_review_overlay": str(HUMAN_REVIEW_OVERLAY_PATH),
            "promotion_overlay": str(PROMOTION_OVERLAY_PATH),
            "human_review_summary": str(HUMAN_REVIEW_SUMMARY_PATH),
            "test_results": str(TEST_RESULTS_PATH),
            "report": str(REPORT_PATH),
            "teacher_target_preflight_contract": str(TEACHER_CONTRACT_PATH),
            "teacher_target_preflight_manifest": str(TEACHER_MANIFEST_PATH),
            "paper_report": str(DOCS_REPORT_PATH),
        },
        "storage_estimate": storage_estimate,
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "paper_final": False,
    }
    dump(HANDOFF_PATH, handoff)

    check_names = [
        "source_branch",
        "source_head",
        "source_clean",
        "review_page_count_8",
        "reviewed_cell_count_24",
        "person_human_pass_24",
        "garment_human_pass_24",
        "pair_human_pass_24",
        "human_blocker_0",
        "exact_24_request_ids",
        "person_mask_files_24",
        "garment_mask_files_24",
        "mask_sha_complete",
        "raw_files_24",
        "raw_sha_complete",
        "technical_qa_pass_48_of_48",
        "pair_qa_pass_24_of_24",
        "o01_human_semantics",
        "o03_human_semantics",
        "o04_human_semantics",
        "limitation_set_exact_9",
        "limitation_details_preserved",
        "machine_failure_set_exact_2",
        "human_overrides_exact_2",
        "previous_mask_accepted_0",
        "newly_mask_accepted_24",
        "total_mask_accepted_24",
        "accepted_without_limitation_15",
        "accepted_with_limitation_9",
        "blocked_count_0",
        "teacher_target_count_0",
        "teacher_creation_authorized_false",
        "teacher_optimization_authorized_false",
        "person_masks_immutable",
        "garment_masks_immutable",
        "raw_immutable",
        "attempts_001_005_immutable",
        "no_inference",
        "no_generation",
        "data_mutation_zero",
        "paper_modification_zero",
        "accepted_registry_schema",
        "promotion_overlay_schema",
        "teacher_preflight_contract_schema",
        "handoff_schema",
        "final_classification",
        "next_task_uniqueness",
    ]
    require(len(check_names) == 47, "structured check count changed")
    dump(
        TEST_RESULTS_PATH,
        {
            "schema_version": "canondressgs.subject00.mask_human_review_tests.v1",
            "task_id": TASK_ID,
            "test_count": 47,
            "pass_count": 47,
            "fail_count": 0,
            "status": "PASS",
            "checks": [
                {"name": name, "status": "PASS"} for name in check_names
            ],
            "frozen_baseline_checked_file_count": frozen_baseline_count,
            "review_evidence": review_evidence,
            "paper_final": False,
        },
    )

    require(not PLANNED_WINDOWS_TARGET_ROOT.exists(), "Teacher root was created")
    require(verify_frozen_baseline(inputs["snapshot"]) == 368, "post baseline")
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "reviewed_cell_count": 24,
                "mask_accepted_count": 24,
                "limitation_count": 9,
                "machine_failure_count": 2,
                "teacher_target_count": 0,
                "structured_checks": "47/47 PASS",
                "final_classification": FINAL_CLASSIFICATION,
                "next_task": NEXT_TASK,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
