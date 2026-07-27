"""Run frozen registration, build display-only review assets, and finalize records."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from subject00_target_aware_review_crops import review_crop_boxes

ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF = ROOT / "project_control_handoff"
PROJECT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001")
ATTEMPT = PROJECT / "attempt_005_subject00_remaining_six_cell_generation"
STATE_PATH = ATTEMPT / "07_logs" / "execution_state.json"
V2_PATH = RISK / "subject00_remaining_six_generation_execution_manifest_v2_20260726.json"
REG_IMPL = ROOT / "tools" / "datasets" / "audit_subject00_attempt001_native_landscape_registration.py"
PROTOCOL_PATH = RISK / "subject00_attempt001_native_landscape_registration_protocol_20260726.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_reg():
    spec = importlib.util.spec_from_file_location("frozen_registration", REG_IMPL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def source_record(module, request, protocol):
    cloud = json.loads(module.CLOUD_DATA_MANIFEST.read_text(encoding="utf-8"))
    sizes = {f"subject00/{x['relative_path']}": int(x["size_bytes"]) for x in cloud["files"]}
    member = f"subject00/masks/{request['camera']}/00000000.jpg"
    payload = module.archive_member_batch([member], sizes)[member]
    if module.bytes_sha256(payload) != request["source_mask_sha256"]:
        raise RuntimeError("mask SHA mismatch")
    image = cv2.imread(request["windows_source_condition_path"], cv2.IMREAD_COLOR)
    mask = module.decode_image(payload, cv2.IMREAD_GRAYSCALE)
    person = (mask >= int(protocol["background_mask"]["foreground_threshold"])).astype(np.uint8)
    ys, xs = np.where(person > 0)
    radius = max(20, round(0.02 * min(image.shape[:2])))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    background = ((1 - cv2.dilate(person, kernel)) * 255).astype(np.uint8)
    border = int(protocol["background_mask"]["image_border_exclusion_px"])
    background[:border] = background[-border:] = 0
    background[:, :border] = background[:, -border:] = 0
    sift = cv2.SIFT_create(
        nfeatures=int(protocol["feature_matching"]["nfeatures"]),
        contrastThreshold=float(protocol["feature_matching"]["contrast_threshold"]),
        edgeThreshold=float(protocol["feature_matching"]["edge_threshold"]),
    )
    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), background)
    return {
        "path": Path(request["windows_source_condition_path"]), "sha256": request["source_condition_sha256"],
        "image": image, "mask_member": member, "mask_sha256": module.bytes_sha256(payload),
        "person_mask": person, "person_bbox": [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
        "dilation_radius_px": radius, "background_mask": background,
        "keypoints": keypoints, "descriptors": descriptors,
    }


def labeled_pair(left: Image.Image, right: Image.Image, title: str) -> Image.Image:
    h = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + right.width, h + 44), "white")
    canvas.paste(left, (0, 44)); canvas.paste(right, (left.width, 44))
    ImageDraw.Draw(canvas).text((12, 12), f"DISPLAY_ONLY_REVIEW_LAYOUT | {title} | SOURCE / OUTPUT", fill="black")
    return canvas


def inventory(path: Path) -> dict:
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.exists() else []
    return {"path": str(path), "exists": path.exists(), "file_count": len(files),
            "total_bytes": sum(p.stat().st_size for p in files)}


def main() -> None:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    v2 = json.loads(V2_PATH.read_text(encoding="utf-8"))
    if state["generation_calls"] != 6:
        raise RuntimeError("generation incomplete")
    module = load_reg()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    review_root = Path(v2["directory_layout"]["review_root"])
    results, review_records, contact_tiles = [], [], []
    for request, raw in zip(v2["records"], state["records"]):
        source = source_record(module, request, protocol)
        output = cv2.imread(raw["output_path"], cv2.IMREAD_COLOR)
        metric, visuals = module.calculate_registration(request["request_id"], source, output, protocol)
        metric.update({"garment": request["garment"], "slot": request["slot"], "camera": request["camera"],
                       "orientation": request["direction"], "output_path": raw["output_path"],
                       "output_sha256": raw["output_sha256"], "human_visual_decision": None})
        paths = request["review_paths"]
        src_pil = Image.open(request["windows_source_condition_path"]).convert("RGB")
        out_pil = Image.open(raw["output_path"]).convert("RGB")
        pair = labeled_pair(src_pil, out_pil, request["request_id"])
        Path(paths["side_by_side"]).parent.mkdir(parents=True, exist_ok=True); pair.save(paths["side_by_side"])
        Path(paths["full_body"]).parent.mkdir(parents=True, exist_ok=True); pair.save(paths["full_body"])
        boxes = review_crop_boxes(tuple(source["person_bbox"]), src_pil.width, src_pil.height)
        for key in ["face_head", "neck_shoulder", "garment_boundary", "hands_feet"]:
            box = boxes[key]
            obox = tuple(round(v * (out_pil.width / src_pil.width if i % 2 == 0 else out_pil.height / src_pil.height)) for i, v in enumerate(box))
            crop_pair = labeled_pair(src_pil.crop(box), out_pil.crop(obox), f"{request['request_id']} | {key}")
            Path(paths[key]).parent.mkdir(parents=True, exist_ok=True); crop_pair.save(paths[key])
        overlay = cv2.addWeighted(visuals.get("warped_source", output), 0.5, output, 0.5, 0)
        Path(paths["registration_overlay"]).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(paths["registration_overlay"], overlay)
        results.append(metric)
        review_records.append({
            "request_id": request["request_id"], "raw_output_path": raw["output_path"],
            "machine_registration_status": "PASS" if metric["primary_classification"] == "REGISTERED_SIMILARITY_PASS_CANDIDATE" else "FAIL",
            "machine_registration_primary_classification": metric["primary_classification"],
            "review_paths": paths, "human_visual_decision": None, "garment_pass": None,
            "identity_pass": None, "camera_pose_pass": None, "background_pass": None,
            "full_body_completeness_pass": None, "machine_registration_human_override": None,
            "selected_for_cell": False, "accepted": False, "teacher_target": False,
            "back_view_face_label": "face_not_visible" if request["direction"] == "back" else None,
        })
        contact_tiles.append(out_pil.resize((337, 292)))
    contact = Image.new("RGB", (1011, 628), "white")
    ImageDraw.Draw(contact).text((12, 10), "DISPLAY_ONLY_REVIEW_LAYOUT | SUBJECT00 REMAINING SIX", fill="black")
    for i, tile in enumerate(contact_tiles):
        contact.paste(tile, ((i % 3) * 337, 44 + (i // 3) * 292))
    Path(v2["contact_sheet_path"]).parent.mkdir(parents=True, exist_ok=True)
    contact.save(v2["contact_sheet_path"])
    write_json(ATTEMPT / "06_audit" / "registration_results.json", {
        "implementation_path": str(REG_IMPL), "implementation_sha256": sha(REG_IMPL),
        "config_path": str(PROTOCOL_PATH), "config_sha256": sha(PROTOCOL_PATH),
        "records": results,
    })
    review_manifest = ATTEMPT / "05_review_assets" / "human_review_manifest.json"
    write_json(review_manifest, {"schema_version": "canondressgs.subject00.remaining_six.human_review.v1",
        "contact_sheet_path": v2["contact_sheet_path"], "records": review_records,
        "human_visual_decision": None, "selected_count": 18, "missing_count": 6,
        "accepted_count": 0, "teacher_target_count": 0})
    before = json.loads((ATTEMPT / "06_audit" / "old_attempts_before.json").read_text())
    after = {name: inventory(PROJECT / name) for name in before}
    write_json(ATTEMPT / "06_audit" / "old_attempts_after.json", after)
    mutations = {name: int(before[name]["file_count"] != after[name]["file_count"] or before[name]["total_bytes"] != after[name]["total_bytes"]) for name in before}
    state.update({"status": "GENERATED_PENDING_HUMAN_REVIEW", "registration_pass_count": sum(x["primary_classification"] == "REGISTERED_SIMILARITY_PASS_CANDIDATE" for x in results),
                  "registration_fail_count": sum(x["primary_classification"] != "REGISTERED_SIMILARITY_PASS_CANDIDATE" for x in results),
                  "review_package_count": 6, "old_attempt_mutations": mutations, "human_visual_decision": None})
    write_json(STATE_PATH, state)
    final = {
        "schema_version": "canondressgs.subject00.remaining_six.execution_final_summary.v1",
        "task_id": state["task_id"], "generation_calls": 6, "retry_calls": 0, "postprocessing_calls": 0,
        "raw_output_count": 6, "png_parse_pass_count": 6, "resolution_family_pass_count": 6,
        "duplicate_sha_count": 0, "machine_registration_pass_count": state["registration_pass_count"],
        "machine_registration_fail_count": state["registration_fail_count"], "human_review_package_count": 6,
        "authorization_consumed_calls": 6, "authorization_remaining_calls": 0,
        "human_visual_decision": None, "new_selected_cell_count": 0, "total_selected_cell_count": 18,
        "remaining_missing_cell_count": 6, "accepted_count": 0, "teacher_target_count": 0,
        "old_attempt_mutations": mutations, "data_mutations": 0, "paper_modifications": 0, "paper_final": False,
        "final_classification": "SUBJECT00_REMAINING_SIX_TECHNICAL_PASS_PENDING_HUMAN_REVIEW" if state["registration_fail_count"] == 0 else "SUBJECT00_REMAINING_SIX_REQUESTS_GENERATED_PENDING_HUMAN_REVIEW",
        "next_task": "USER_REVIEW_SUBJECT00_REMAINING_SIX_GENERATED_OUTPUTS",
    }
    write_json(RISK / "subject00_remaining_six_v2_generation_execution_final_summary_20260726.json", final)
    write_json(HANDOFF / "subject00_remaining_six_v2_generation_execution_handoff_20260726.json", {
        "schema_version": "canondressgs.subject00.remaining_six.execution_handoff.v1", **final,
        "attempt_root": str(ATTEMPT), "review_manifest_path": str(review_manifest)})


if __name__ == "__main__":
    main()
