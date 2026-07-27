#!/usr/bin/env python3
"""Manage Subject00 O03 canary native-resolution correction continuation.

This script does not call any image-generation API. Codex invokes the
platform-managed imagegen tool directly, then this script records and validates
the returned PNGs without resizing, cropping, padding, re-encoding, or repair.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF = ROOT / "project_control_handoff"

TASK_ID = "AAAI27-SUBJECT00-O03-CANARY-RESOLUTION-CORRECTION-AND-CONTINUATION-001"
PRIOR_TASK_ID = "AAAI27-SUBJECT00-O03-FROZEN-CONTRACT-CANARY-EXECUTION-001"
SOURCE_BRANCH = "research/subject00-o03-frozen-contract-canary-execution-20260726"
SOURCE_HEAD = "ba1043e8eeba1690ccf64c8f96b47c58425acbb9"
SOURCE_WORKTREE = Path(r"E:\model_train\canondressgs_subject00_o03_frozen_contract_canary_execution")
NEW_BRANCH = "research/subject00-o03-canary-resolution-correction-continuation-20260726"
ATTEMPT_NAMESPACE = "attempt_004_o03_hood_removal_targeted_canary"
AUTHORIZATION_SCOPE = "O03_FROZEN_TARGETED_CANARY_4_REQUESTS_ONLY"

OLD_EXACT_SIZE = (1349, 1166)
ACCEPTED_SIZES = ((1348, 1167), (1349, 1166), (1350, 1165))
ACCEPTED_FAMILY_NAME = "SUBJECT00_NATIVE_LANDSCAPE_2515_FAMILY"
CORRECTION_REASON = "PLATFORM_MANAGED_IMAGE_GENERATION_DOES_NOT_GUARANTEE_EXACT_SINGLE_PIXEL_DIMENSIONS"

EXPECTED_REQUEST_IDS = [
    "subject00_O03_slot00_canary_attempt004_cand00",
    "subject00_O03_slot04_canary_attempt004_cand00",
    "subject00_O03_slot05_canary_attempt004_cand00",
    "subject00_O03_slot07_canary_attempt004_cand00",
]
FIRST_REQUEST_ID = EXPECTED_REQUEST_IDS[0]
FIRST_EXPECTED_SHA = "f5197d9646ccf60ddffe3184e295e2afed2f04ff3ccbe95b5908974f6652e9e1"
FIRST_EXPECTED_SIZE = (1350, 1165)

DATA_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001")
ATTEMPT_ROOT = DATA_ROOT / ATTEMPT_NAMESPACE
FIRST_RAW_PATH = ATTEMPT_ROOT / "04_generation_responses" / "technical_failures" / f"{FIRST_REQUEST_ID}.png"
CONTINUATION_STATE_PATH = ATTEMPT_ROOT / "06_provenance" / "correction_continuation_manifest.json"
FIRST_RECLASSIFICATION_PATH = (
    ATTEMPT_ROOT / "06_provenance" / FIRST_REQUEST_ID / "corrected_resolution_reclassification_sidecar.json"
)
RAW_REGISTRY_EXTERNAL_PATH = ATTEMPT_ROOT / "08_technical_validation" / "corrected_raw_output_registry.json"
REGISTRATION_RESULTS_EXTERNAL_PATH = ATTEMPT_ROOT / "08_technical_validation" / "corrected_registration_results.json"
REVIEW_MANIFEST_EXTERNAL_PATH = ATTEMPT_ROOT / "07_human_review" / "correction_continuation_human_review_manifest.json"
FINAL_SUMMARY_EXTERNAL_PATH = ATTEMPT_ROOT / "09_final" / "correction_continuation_final_summary.json"

FROZEN_CONTRACT_REL = Path("paper_protocol/reviewer_risk/SUBJECT00_O03_HOOD_REMOVAL_CANARY_CONTRACT_20260726.md")
FROZEN_MANIFEST_REL = Path("paper_protocol/reviewer_risk/subject00_o03_hood_removal_canary_manifest_draft_20260726.json")
OVERLAY_REL = Path("paper_protocol/reviewer_risk/subject00_o03_canary_native_resolution_correction_overlay_20260726.json")
CORRECTION_REPORT_REL = Path("paper_protocol/reviewer_risk/SUBJECT00_O03_CANARY_NATIVE_RESOLUTION_CORRECTION_20260726.md")
CONTINUATION_MANIFEST_REL = Path("paper_protocol/reviewer_risk/subject00_o03_canary_resolution_correction_continuation_manifest_20260726.json")
RAW_REGISTRY_REL = Path("paper_protocol/reviewer_risk/subject00_o03_canary_corrected_raw_output_registry_20260726.json")
REGISTRATION_RESULTS_REL = Path("paper_protocol/reviewer_risk/subject00_o03_canary_corrected_registration_results_20260726.json")
REVIEW_MANIFEST_REL = Path("paper_protocol/reviewer_risk/subject00_o03_canary_resolution_correction_human_review_manifest_20260726.json")
FINAL_SUMMARY_REL = Path("paper_protocol/reviewer_risk/subject00_o03_canary_resolution_correction_final_summary_20260726.json")
TESTS_REL = Path("paper_protocol/reviewer_risk/subject00_o03_canary_resolution_correction_continuation_tests_20260726.json")
HANDOFF_REL = Path("project_control_handoff/subject00_o03_canary_resolution_correction_continuation_handoff_20260726.json")

ATTEMPT001_AUDIT_WORKTREE = Path(r"E:\model_train\canondressgs_subject00_attempt001_native_landscape_registration_audit")
ATTEMPT001_AUDIT_SCRIPT_REL = Path("tools/datasets/audit_subject00_attempt001_native_landscape_registration.py")
ATTEMPT001_CHECKER_REL = Path("tools/datasets/check_subject00_attempt001_native_landscape_registration.py")
ATTEMPT001_PROTOCOL_EXTERNAL = (
    DATA_ROOT / "attempt_001_native_landscape_registration_audit" / "00_provenance" / "registration_audit_protocol.json"
)
ATTEMPT001_TESTS_EXTERNAL = (
    DATA_ROOT / "attempt_001_native_landscape_registration_audit" / "07_final_summary" / "attempt001_native_landscape_registration_tests.json"
)
ATTEMPT001_HANDOFF_REL = Path("project_control_handoff/subject00_attempt001_native_landscape_registration_audit_handoff_20260726.json")
ATTEMPT001_PROTOCOL_RECORDED_SHA = "f8bda763ee845d7e3fd6ece32d7b4a7e8150308cd69df0aa80f15cd69aa53e76"

PROVIDER = "CODEX_IMAGE_GENERATION_SKILL"
MODE = "CODEX_PLATFORM_MANAGED_DIRECT_IMAGE_EDIT"
TOOL_NAME = "image_gen.imagegen"
SKILL_NAME = "imagegen"

OLD_ATTEMPTS = {
    "attempt_001": DATA_ROOT / "attempt_001",
    "attempt_002": DATA_ROOT / "attempt_002_portrait_canary",
    "attempt_003": DATA_ROOT / "attempt_003_portrait_canary_v2_registered_outpaint",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    if isinstance(value, dict):
        value = dict(value)
        value.pop("content_sha256", None)
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        payload = dict(value)
        payload.pop("content_sha256", None)
        payload["content_sha256"] = canonical_sha256(payload)
    else:
        payload = value
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value.rstrip() + "\n", encoding="utf-8")
    tmp.replace(path)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8").strip()


def image_meta(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        return {
            "path": str(path),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "png_parse_status": "PASS" if image.format == "PNG" else "FAIL",
        }


def normalized_size(meta: dict[str, Any]) -> str:
    return f"{meta['width']}x{meta['height']}"


def family_pass(meta: dict[str, Any]) -> bool:
    return (int(meta["width"]), int(meta["height"])) in ACCEPTED_SIZES and meta.get("format") == "PNG"


def tree_snapshot(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"root": str(root), "exists": False, "file_count": 0, "total_bytes": 0, "tree_sha256": None}
    digest = hashlib.sha256()
    total = 0
    count = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        sha = file_sha256(path)
        count += 1
        total += size
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha.encode("ascii"))
        digest.update(b"\n")
    return {"root": str(root), "exists": True, "file_count": count, "total_bytes": total, "tree_sha256": digest.hexdigest()}


def compare_snapshot(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name in OLD_ATTEMPTS:
        b = before[name]
        a = after[name]
        mutated = (
            b["exists"] != a["exists"]
            or b["file_count"] != a["file_count"]
            or b["total_bytes"] != a["total_bytes"]
            or b["tree_sha256"] != a["tree_sha256"]
        )
        result[name] = {
            "mutations": 1 if mutated else 0,
            "before": b,
            "after": a,
        }
    return result


def current_old_attempts() -> dict[str, Any]:
    return {name: tree_snapshot(path) for name, path in OLD_ATTEMPTS.items()}


def old_attempt_comparison() -> dict[str, Any]:
    before_path = ATTEMPT_ROOT / "06_provenance" / "old_attempt_snapshot_before.json"
    before = read_json(before_path)
    return compare_snapshot(before, current_old_attempts())


def load_registration_module() -> Any:
    script = ROOT / ATTEMPT001_AUDIT_SCRIPT_REL
    if not script.is_file():
        raise FileNotFoundError(script)
    spec = importlib.util.spec_from_file_location("attempt001_native_landscape_registration", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load registration implementation: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_frozen_records() -> list[dict[str, Any]]:
    manifest = read_json(ATTEMPT_ROOT / "06_provenance" / "generation_manifest.json")
    if manifest["task_id"] != PRIOR_TASK_ID:
        raise RuntimeError("prior generation manifest task mismatch")
    if manifest["attempt_namespace"] != ATTEMPT_NAMESPACE:
        raise RuntimeError("attempt namespace mismatch")
    ids = [record["request_id"] for record in manifest["records"]]
    if ids != EXPECTED_REQUEST_IDS:
        raise RuntimeError(f"request ID drift: {ids!r}")
    return manifest["records"]


def validate_preflight() -> dict[str, Any]:
    facts = {
        "source_branch": git(SOURCE_WORKTREE, "branch", "--show-current"),
        "source_head": git(SOURCE_WORKTREE, "rev-parse", "HEAD"),
        "source_status_short": git(SOURCE_WORKTREE, "status", "--short"),
        "source_status_porcelain_v2": git(SOURCE_WORKTREE, "status", "--porcelain=v2"),
        "new_branch": git(ROOT, "branch", "--show-current"),
        "new_head": git(ROOT, "rev-parse", "HEAD"),
        "new_status_short": git(ROOT, "status", "--short"),
    }
    expected = {
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "source_status_short": "",
        "source_status_porcelain_v2": "",
        "new_branch": NEW_BRANCH,
        "new_head": SOURCE_HEAD,
    }
    for key, value in expected.items():
        if facts[key] != value:
            raise RuntimeError(f"preflight mismatch {key}: {facts[key]!r} != {value!r}")

    records = load_frozen_records()
    calls = sum(int(record.get("generation_calls", 0)) for record in records)
    if calls != 1:
        raise RuntimeError(f"generation_calls_so_far mismatch: {calls}")
    first = records[0]
    if first["request_id"] != FIRST_REQUEST_ID or first.get("status") != "TECHNICAL_FAILURE":
        raise RuntimeError("first request status mismatch")
    meta = image_meta(FIRST_RAW_PATH)
    if meta["sha256"] != FIRST_EXPECTED_SHA:
        raise RuntimeError("first raw SHA mismatch")
    if (meta["width"], meta["height"]) != FIRST_EXPECTED_SIZE:
        raise RuntimeError("first raw resolution mismatch")
    comparison = old_attempt_comparison()
    if any(item["mutations"] for item in comparison.values()):
        raise RuntimeError(f"old attempt mutation detected: {comparison!r}")

    protocol = read_json(ATTEMPT001_PROTOCOL_EXTERNAL)
    tests = read_json(ATTEMPT001_TESTS_EXTERNAL)
    if protocol.get("protocol_file_sha256") != ATTEMPT001_PROTOCOL_RECORDED_SHA:
        raise RuntimeError("attempt001 protocol recorded SHA mismatch")
    if tests.get("result") != "PASS" or tests.get("test_count") != 79:
        raise RuntimeError("attempt001 registration tests are not the frozen PASS/79 contract")
    return {
        "status": "PASS",
        "facts": facts,
        "generation_calls_so_far": calls,
        "first_raw_artifact": meta,
        "old_attempt_comparison": comparison,
        "registration_contract": registration_contract_record(),
    }


def registration_contract_record() -> dict[str, Any]:
    local_script = ROOT / ATTEMPT001_AUDIT_SCRIPT_REL
    source_script = ATTEMPT001_AUDIT_WORKTREE / ATTEMPT001_AUDIT_SCRIPT_REL
    checker = ATTEMPT001_AUDIT_WORKTREE / ATTEMPT001_CHECKER_REL
    handoff = ROOT / ATTEMPT001_HANDOFF_REL
    return {
        "status": "RECOVERED_UNIQUELY",
        "implementation_path_used": str(local_script),
        "implementation_sha256_used": file_sha256(local_script),
        "attempt001_worktree_implementation_path": str(source_script),
        "attempt001_worktree_implementation_sha256": file_sha256(source_script) if source_script.is_file() else None,
        "attempt001_checker_path": str(checker),
        "attempt001_checker_sha256": file_sha256(checker) if checker.is_file() else None,
        "protocol_path": str(ATTEMPT001_PROTOCOL_EXTERNAL),
        "protocol_file_sha256_recorded": ATTEMPT001_PROTOCOL_RECORDED_SHA,
        "protocol_actual_file_sha256": file_sha256(ATTEMPT001_PROTOCOL_EXTERNAL),
        "tests_path": str(ATTEMPT001_TESTS_EXTERNAL),
        "tests_actual_file_sha256": file_sha256(ATTEMPT001_TESTS_EXTERNAL),
        "handoff_path": str(handoff),
        "handoff_sha256": file_sha256(handoff) if handoff.is_file() else None,
        "runtime_python": "D:\\miniconda3\\envs\\garment-mask-v3a\\python.exe",
    }


def write_correction_overlay() -> dict[str, Any]:
    preflight = validate_preflight()
    records = load_frozen_records()
    source_condition_paths = [record["source_condition_path"] for record in records]
    source_condition_sha256 = [record["source_condition_sha256"] for record in records]
    overlay = {
        "schema_version": "canondressgs.subject00.o03_canary.native_resolution_correction_overlay.v1",
        "task_id": TASK_ID,
        "prior_task_id": PRIOR_TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "attempt_namespace": ATTEMPT_NAMESPACE,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "correction_reason": CORRECTION_REASON,
        "original_exact_raw_resolution_required": {"width": OLD_EXACT_SIZE[0], "height": OLD_EXACT_SIZE[1]},
        "corrected_raw_native_landscape_resolution_family_name": ACCEPTED_FAMILY_NAME,
        "corrected_raw_native_landscape_resolution_accepted_set": [
            {"width": width, "height": height, "width_plus_height": width + height}
            for width, height in ACCEPTED_SIZES
        ],
        "scope_limited_to_raw_size_acceptance": True,
        "unchanged_contract_surfaces": [
            "request_ids",
            "source_conditions",
            "prompt",
            "generation_method",
            "candidate_count",
            "retry_policy",
            "postprocessing_policy",
            "human_review_required",
            "accepted_false",
            "teacher_target_false",
        ],
        "forbidden_expansions": [
            "portrait_resolution",
            "arbitrary_width_height",
            "resized_result",
            "outpaint_result",
            "padded_canvas",
            "cropped_result",
        ],
        "first_raw_artifact_preservation": {
            "must_not_move_rename_overwrite_or_regenerate": True,
            "path": str(FIRST_RAW_PATH),
            "sha256": FIRST_EXPECTED_SHA,
            "resolution": f"{FIRST_EXPECTED_SIZE[0]}x{FIRST_EXPECTED_SIZE[1]}",
        },
        "source_condition_paths": source_condition_paths,
        "source_condition_sha256": source_condition_sha256,
        "registration_contract": registration_contract_record(),
        "paper_final": False,
    }
    write_json(ROOT / OVERLAY_REL, overlay)

    report = [
        "# Subject00 O03 Canary Native Resolution Correction",
        "",
        f"- Task ID: `{TASK_ID}`",
        f"- Source branch/head: `{SOURCE_BRANCH}` / `{SOURCE_HEAD}`",
        f"- New branch: `{NEW_BRANCH}`",
        f"- Attempt namespace: `{ATTEMPT_NAMESPACE}`",
        f"- Correction reason: `{CORRECTION_REASON}`",
        f"- Old exact gate: `{OLD_EXACT_SIZE[0]}x{OLD_EXACT_SIZE[1]}`",
        f"- Corrected family: `{ACCEPTED_FAMILY_NAME}` = `1348x1167`, `1349x1166`, `1350x1165`",
        f"- First raw artifact remains at: `{FIRST_RAW_PATH}`",
        "",
        "This overlay changes only raw native size admissibility. It does not change request IDs, source bindings, prompt text, generation method, retry policy, postprocessing policy, human-review policy, or accepted/Teacher-target state.",
        "",
        "The attempt001 native-landscape registration implementation and frozen thresholds are reused for machine registration. Machine registration is not a human visual decision.",
    ]
    write_text(ROOT / CORRECTION_REPORT_REL, "\n".join(report))

    if not CONTINUATION_STATE_PATH.exists():
        state = initialize_state(preflight)
        write_json(CONTINUATION_STATE_PATH, state)
    return {"status": "PASS", "overlay": str(ROOT / OVERLAY_REL), "report": str(ROOT / CORRECTION_REPORT_REL)}


def initialize_state(preflight: dict[str, Any]) -> dict[str, Any]:
    records = []
    for record in load_frozen_records():
        item = {
            "request_id": record["request_id"],
            "source_request_id": record["source_request_id"],
            "garment": record["garment"],
            "slot": record["slot"],
            "camera": record["camera"],
            "direction": record["direction"],
            "source_condition_path": record["source_condition_path"],
            "source_condition_sha256": record["source_condition_sha256"],
            "source_request_path": record["source_request_path"],
            "source_request_sha256": record["source_request_sha256"],
            "prompt_path": record["prompt_path"],
            "prompt_sha256": record["prompt_sha256"],
            "output_path_contract": record["output_path"],
            "generation_record_path": record["generation_record_path"],
            "original_execution_status": record["status"],
            "retry_count": 0,
            "postprocessing_calls": 0,
            "human_visual_decision": None,
            "accepted": False,
            "teacher_target": False,
        }
        if record["request_id"] == FIRST_REQUEST_ID:
            first_meta = image_meta(FIRST_RAW_PATH)
            item.update(
                {
                    "raw_output_path": str(FIRST_RAW_PATH),
                    "raw_output_sha256": first_meta["sha256"],
                    "raw_output_resolution": normalized_size(first_meta),
                    "png_parse_status": first_meta["png_parse_status"],
                    "resolution_family_pass": family_pass(first_meta),
                    "original_execution_status": "NATIVE_RESOLUTION_MISMATCH_UNDER_OLD_EXACT_GATE",
                    "corrected_technical_status": "RAW_NATIVE_RESOLUTION_FAMILY_PASS_PENDING_REGISTRATION_AND_HUMAN_REVIEW",
                    "generation_call_index": 1,
                    "additional_generation_call": False,
                }
            )
        else:
            item.update(
                {
                    "raw_output_path": None,
                    "raw_output_sha256": None,
                    "raw_output_resolution": None,
                    "png_parse_status": None,
                    "resolution_family_pass": None,
                    "corrected_technical_status": "PENDING_GENERATION",
                    "generation_call_index": None,
                    "additional_generation_call": None,
                }
            )
        records.append(item)
    return {
        "schema_version": "canondressgs.subject00.o03_canary.resolution_correction_continuation_manifest.v1",
        "task_id": TASK_ID,
        "prior_task_id": PRIOR_TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "new_worktree": str(ROOT),
        "attempt_namespace": ATTEMPT_NAMESPACE,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "generation_calls_before_task": 1,
        "additional_generation_calls": 0,
        "total_generation_calls": 1,
        "total_authorized_generation_calls": 4,
        "retry_calls": 0,
        "postprocessing_calls": 0,
        "raw_native_resolution_accepted_set": [f"{w}x{h}" for w, h in ACCEPTED_SIZES],
        "preflight": preflight,
        "registration_contract": registration_contract_record(),
        "records": records,
        "status": "CORRECTION_OVERLAY_WRITTEN_PENDING_FIRST_REGISTRATION",
        "paper_final": False,
    }


def source_cache_for_records(records: list[dict[str, Any]], protocol: dict[str, Any], module: Any) -> dict[str, dict[str, Any]]:
    cloud_manifest = read_json(module.CLOUD_DATA_MANIFEST)
    archive_sizes = {f"subject00/{item['relative_path']}": int(item["size_bytes"]) for item in cloud_manifest["files"]}
    mask_members = sorted({f"subject00/masks/{record['camera']}/00000000.jpg" for record in records})
    mask_payloads = module.archive_member_batch(mask_members, archive_sizes)
    source_cache: dict[str, dict[str, Any]] = {}
    for record in records:
        slot = record["slot"]
        if slot in source_cache:
            continue
        source_path = Path(record["source_condition_path"])
        if file_sha256(source_path) != record["source_condition_sha256"]:
            raise RuntimeError(f"source condition SHA mismatch: {record['request_id']}")
        mask_member = f"subject00/masks/{record['camera']}/00000000.jpg"
        mask_bytes = mask_payloads[mask_member]
        manifest = read_json(ROOT / FROZEN_MANIFEST_REL)
        request_binding = {item["request_id"]: item for item in manifest["requests"]}[record["request_id"]]
        if module.bytes_sha256(mask_bytes) != request_binding["source_mask_sha256"]:
            raise RuntimeError(f"source mask SHA mismatch: {record['request_id']}")
        image = module.cv2.imread(str(source_path), module.cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"source image OpenCV parse failed: {source_path}")
        mask_image = module.decode_image(mask_bytes, module.cv2.IMREAD_GRAYSCALE)
        person_mask = (mask_image >= int(protocol["background_mask"]["foreground_threshold"])).astype(np.uint8)
        ys, xs = np.where(person_mask > 0)
        if not len(xs):
            raise RuntimeError(f"empty person mask: {record['request_id']}")
        radius = max(20, round(0.02 * min(image.shape[:2])))
        expected_radius = int(protocol["background_mask"]["person_dilation_radius_px_for_1330x1150"])
        if radius != expected_radius:
            raise RuntimeError(f"person dilation changed: {radius}")
        kernel = module.cv2.getStructuringElement(module.cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        exclusion = module.cv2.dilate(person_mask, kernel)
        background_mask = ((1 - exclusion) * 255).astype(np.uint8)
        border = int(protocol["background_mask"]["image_border_exclusion_px"])
        background_mask[:border] = 0
        background_mask[-border:] = 0
        background_mask[:, :border] = 0
        background_mask[:, -border:] = 0
        sift = module.cv2.SIFT_create(
            nfeatures=int(protocol["feature_matching"]["nfeatures"]),
            contrastThreshold=float(protocol["feature_matching"]["contrast_threshold"]),
            edgeThreshold=float(protocol["feature_matching"]["edge_threshold"]),
        )
        keypoints, descriptors = sift.detectAndCompute(module.cv2.cvtColor(image, module.cv2.COLOR_BGR2GRAY), background_mask)
        source_cache[slot] = {
            "path": source_path,
            "sha256": file_sha256(source_path),
            "image": image,
            "mask_member": mask_member,
            "mask_sha256": module.bytes_sha256(mask_bytes),
            "person_mask": person_mask,
            "person_bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
            "dilation_radius_px": radius,
            "background_mask": background_mask,
            "keypoints": keypoints,
            "descriptors": descriptors,
        }
    return source_cache


def run_registration(record: dict[str, Any], output_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    module = load_registration_module()
    protocol = read_json(ATTEMPT001_PROTOCOL_EXTERNAL)
    if protocol.get("protocol_file_sha256") != ATTEMPT001_PROTOCOL_RECORDED_SHA:
        raise RuntimeError("registration protocol SHA mismatch")
    state_records = load_state()["records"] if CONTINUATION_STATE_PATH.exists() else load_frozen_records()
    source_cache = source_cache_for_records(state_records, protocol, module)
    output = module.cv2.imread(str(output_path), module.cv2.IMREAD_COLOR)
    if output is None:
        raise RuntimeError(f"OpenCV output parse failed: {output_path}")
    metric, visuals = module.calculate_registration(record["request_id"], source_cache[record["slot"]], output, protocol)
    metric.update(
        {
            "garment": record["garment"],
            "slot": record["slot"],
            "camera": record["camera"],
            "orientation": record["direction"],
            "output_path": str(output_path),
            "output_sha256": file_sha256(output_path),
            "protocol_sha256_recorded": ATTEMPT001_PROTOCOL_RECORDED_SHA,
            "human_visual_decision": None,
        }
    )
    return metric, visuals


def load_state() -> dict[str, Any]:
    if not CONTINUATION_STATE_PATH.exists():
        raise FileNotFoundError(CONTINUATION_STATE_PATH)
    return read_json(CONTINUATION_STATE_PATH)


def save_state(state: dict[str, Any]) -> None:
    write_json(CONTINUATION_STATE_PATH, state)


def record_index(state: dict[str, Any], request_id: str) -> int:
    matches = [idx for idx, record in enumerate(state["records"]) if record["request_id"] == request_id]
    if len(matches) != 1:
        raise RuntimeError(f"unknown request id: {request_id}")
    return matches[0]


def write_registration_payload(state: dict[str, Any]) -> None:
    metrics = [record["registration_metric"] for record in state["records"] if record.get("registration_metric")]
    payload = {
        "schema_version": "canondressgs.subject00.o03_canary.corrected_registration_results.v1",
        "task_id": TASK_ID,
        "attempt_namespace": ATTEMPT_NAMESPACE,
        "registration_contract": registration_contract_record(),
        "audit_count": len(metrics),
        "machine_registration_pass_count": sum(
            item["primary_classification"] == "REGISTERED_SIMILARITY_PASS_CANDIDATE" for item in metrics
        ),
        "classification_counts": dict(sorted(Counter(item["primary_classification"] for item in metrics).items())),
        "records": metrics,
        "paper_final": False,
    }
    write_json(REGISTRATION_RESULTS_EXTERNAL_PATH, payload)
    write_json(ROOT / REGISTRATION_RESULTS_REL, payload)


def check_first() -> dict[str, Any]:
    if not CONTINUATION_STATE_PATH.exists():
        write_correction_overlay()
    state = load_state()
    idx = record_index(state, FIRST_REQUEST_ID)
    record = state["records"][idx]
    meta = image_meta(FIRST_RAW_PATH)
    if meta["sha256"] != FIRST_EXPECTED_SHA or (meta["width"], meta["height"]) != FIRST_EXPECTED_SIZE:
        raise RuntimeError("first raw artifact changed")
    if not family_pass(meta):
        raise RuntimeError("first raw artifact does not pass corrected family")
    metric, visuals = run_registration(record, FIRST_RAW_PATH)
    primary = metric["primary_classification"]
    machine_pass = primary == "REGISTERED_SIMILARITY_PASS_CANDIDATE"
    record.update(
        {
            "raw_output_path": str(FIRST_RAW_PATH),
            "raw_output_sha256": meta["sha256"],
            "raw_output_resolution": normalized_size(meta),
            "png_parse_status": "PASS",
            "resolution_family_pass": True,
            "sha_unique": True,
            "source_binding_status": "PASS",
            "registration_metric": metric,
            "machine_registration_status": "PASS" if machine_pass else "FAIL",
            "machine_registration_primary_classification": primary,
            "first_output_technical_admissibility": (
                "PASS_PENDING_HUMAN_REVIEW" if machine_pass else "MACHINE_REGISTRATION_FAIL_PENDING_OPTIONAL_HUMAN_INSPECTION"
            ),
            "human_review_eligible": machine_pass,
        }
    )
    save_review_visuals(record, FIRST_RAW_PATH, visuals)
    state["records"][idx] = record
    state["status"] = "FIRST_OUTPUT_CHECKED_CONTINUE_REMAINING_REQUESTS"
    update_state_counts(state)
    save_state(state)
    sidecar = {
        "schema_version": "canondressgs.subject00.o03_canary.first_output_resolution_reclassification.v1",
        "task_id": TASK_ID,
        "request_id": FIRST_REQUEST_ID,
        "raw_output_path": str(FIRST_RAW_PATH),
        "raw_output_sha256": meta["sha256"],
        "raw_output_resolution": normalized_size(meta),
        "original_execution_status": "NATIVE_RESOLUTION_MISMATCH_UNDER_OLD_EXACT_GATE",
        "corrected_technical_status": "RAW_NATIVE_RESOLUTION_FAMILY_PASS_PENDING_REGISTRATION_AND_HUMAN_REVIEW",
        "first_output_technical_admissibility": record["first_output_technical_admissibility"],
        "machine_registration_primary_classification": primary,
        "retry_count": 0,
        "postprocessing_calls": 0,
        "raw_file_moved_renamed_overwritten_regenerated": False,
        "human_visual_decision": None,
        "accepted": False,
        "teacher_target": False,
    }
    write_json(FIRST_RECLASSIFICATION_PATH, sidecar)
    write_registration_payload(state)
    return {
        "status": record["first_output_technical_admissibility"],
        "request_id": FIRST_REQUEST_ID,
        "resolution": normalized_size(meta),
        "sha256": meta["sha256"],
        "machine_registration_primary_classification": primary,
    }


def raw_output_shas(state: dict[str, Any]) -> set[str]:
    return {
        str(record["raw_output_sha256"])
        for record in state["records"]
        if record.get("raw_output_sha256")
    }


def update_state_counts(state: dict[str, Any]) -> None:
    records = state["records"]
    raw = [record for record in records if record.get("raw_output_path")]
    state["raw_generated_output_count"] = len(raw)
    state["png_parse_pass_count"] = sum(record.get("png_parse_status") == "PASS" for record in raw)
    state["resolution_family_pass_count"] = sum(bool(record.get("resolution_family_pass")) for record in raw)
    state["machine_registration_pass_count"] = sum(record.get("machine_registration_status") == "PASS" for record in raw)
    state["human_review_eligible_count"] = sum(bool(record.get("human_review_eligible")) for record in raw)
    state["raw_output_paths"] = [record["raw_output_path"] for record in raw]
    state["raw_output_resolutions"] = [record["raw_output_resolution"] for record in raw]
    state["raw_output_sha256"] = [record["raw_output_sha256"] for record in raw]


def record_generated_output(request_id: str, tool_output: Path, tool_output_hint: str | None) -> dict[str, Any]:
    state = load_state()
    if not state["records"][0].get("registration_metric"):
        raise RuntimeError("first output registration must be checked before continuing generation")
    idx = record_index(state, request_id)
    if idx == 0:
        raise RuntimeError("first output cannot be regenerated or re-recorded")
    record = state["records"][idx]
    if record.get("raw_output_path"):
        raise RuntimeError(f"request already has raw output: {request_id}")
    if state["additional_generation_calls"] >= 3 or state["total_generation_calls"] >= 4:
        raise RuntimeError("generation budget exhausted")

    meta = image_meta(tool_output)
    previous_shas = raw_output_shas(state)
    duplicate = meta["sha256"] in previous_shas
    pass_family = family_pass(meta)
    destination = Path(record["output_path_contract"]) if pass_family and not duplicate else (
        ATTEMPT_ROOT / "04_generation_responses" / "technical_failures" / f"{request_id}.png"
    )
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(tool_output, tmp)
    if file_sha256(tmp) != meta["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("raw copy SHA mismatch")
    tmp.replace(destination)
    copied = image_meta(destination)
    record.update(
        {
            "raw_output_path": str(destination),
            "raw_output_sha256": copied["sha256"],
            "raw_output_resolution": normalized_size(copied),
            "png_parse_status": copied["png_parse_status"],
            "resolution_family_pass": pass_family,
            "sha_unique": not duplicate,
            "source_binding_status": "PASS",
            "generation_call_index": state["total_generation_calls"] + 1,
            "additional_generation_call": True,
            "tool_output_hint": tool_output_hint,
        }
    )
    state["additional_generation_calls"] += 1
    state["total_generation_calls"] += 1

    if not pass_family:
        record["corrected_technical_status"] = "RAW_NATIVE_RESOLUTION_FAMILY_FAIL_TECHNICAL_FAILURE_NO_RETRY"
        record["machine_registration_status"] = "NOT_RUN_NON_WHITELIST_RESOLUTION"
        record["human_review_eligible"] = False
        state["status"] = "PARTIAL_STOPPED_NON_WHITELIST_RESOLUTION"
    elif duplicate:
        record["corrected_technical_status"] = "DUPLICATE_SHA_TECHNICAL_FAILURE_NO_RETRY"
        record["machine_registration_status"] = "NOT_RUN_DUPLICATE_SHA"
        record["human_review_eligible"] = False
        state["status"] = "PARTIAL_STOPPED_DUPLICATE_SHA"
    else:
        metric, visuals = run_registration(record, destination)
        primary = metric["primary_classification"]
        machine_pass = primary == "REGISTERED_SIMILARITY_PASS_CANDIDATE"
        record.update(
            {
                "corrected_technical_status": (
                    "RAW_NATIVE_RESOLUTION_FAMILY_PASS_MACHINE_REGISTRATION_PASS_PENDING_HUMAN_REVIEW"
                    if machine_pass
                    else "RAW_NATIVE_RESOLUTION_FAMILY_PASS_MACHINE_REGISTRATION_FAIL_PENDING_OPTIONAL_HUMAN_INSPECTION"
                ),
                "registration_metric": metric,
                "machine_registration_status": "PASS" if machine_pass else "FAIL",
                "machine_registration_primary_classification": primary,
                "human_review_eligible": machine_pass,
            }
        )
        save_review_visuals(record, destination, visuals)
        state["status"] = "GENERATION_CONTINUATION_IN_PROGRESS"

    state["records"][idx] = record
    update_state_counts(state)
    save_state(state)
    write_registration_payload(state)
    write_generation_record(record)
    return {
        "status": record["corrected_technical_status"],
        "request_id": request_id,
        "raw_output_path": record["raw_output_path"],
        "resolution": record["raw_output_resolution"],
        "sha256": record["raw_output_sha256"],
        "generation_calls": state["total_generation_calls"],
        "continue_generation": bool(pass_family and not duplicate),
        "machine_registration_primary_classification": record.get("machine_registration_primary_classification"),
    }


def write_generation_record(record: dict[str, Any]) -> None:
    payload = {
        "schema_version": "canondressgs.subject00.o03_canary.correction_continuation_generation_record.v1",
        "task_id": TASK_ID,
        "request_id": record["request_id"],
        "generation_provider": PROVIDER,
        "generation_mode": MODE,
        "skill_name": SKILL_NAME,
        "tool_name": TOOL_NAME,
        "source_condition_path": record["source_condition_path"],
        "source_condition_sha256": record["source_condition_sha256"],
        "prompt_path": record["prompt_path"],
        "prompt_sha256": record["prompt_sha256"],
        "raw_output_path": record["raw_output_path"],
        "raw_output_sha256": record["raw_output_sha256"],
        "raw_output_resolution": record["raw_output_resolution"],
        "resolution_family_pass": record["resolution_family_pass"],
        "sha_unique": record["sha_unique"],
        "machine_registration_status": record.get("machine_registration_status"),
        "machine_registration_primary_classification": record.get("machine_registration_primary_classification"),
        "retry_count": 0,
        "postprocessing_calls": 0,
        "raw_output_immutable": True,
        "human_visual_decision": None,
        "accepted": False,
        "teacher_target": False,
        "recorded_at": now(),
    }
    write_json(Path(record["generation_record_path"]).with_name("correction_continuation_generation_record.json"), payload)


def pair_canvas(source_path: Path, output_path: Path, label: str, crop: tuple[float, float, float, float] | None = None) -> Image.Image:
    with Image.open(source_path) as src, Image.open(output_path) as out:
        src_rgb = src.convert("RGB")
        out_rgb = out.convert("RGB")
        if crop is not None:
            src_rgb = src_rgb.crop(scale_box(src_rgb.size, crop))
            out_rgb = out_rgb.crop(scale_box(out_rgb.size, crop))
        target_h = 620
        src_tile = fit_pil(src_rgb, 620, target_h)
        out_tile = fit_pil(out_rgb, 728, target_h)
        label_h = 58
        canvas = Image.new("RGB", (src_tile.width + out_tile.width, target_h + label_h), "white")
        canvas.paste(src_tile, (0, label_h))
        canvas.paste(out_tile, (src_tile.width, label_h))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        draw.text((10, 8), f"{label} | DISPLAY_ONLY_REVIEW_LAYOUT", fill="black", font=font)
        draw.text((10, 31), "SOURCE CONDITION", fill="black", font=font)
        draw.text((src_tile.width + 10, 31), "RAW GENERATED OUTPUT", fill="black", font=font)
        return canvas


def scale_box(size: tuple[int, int], fractions: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    w, h = size
    l, t, r, b = fractions
    return (int(w * l), int(h * t), int(w * r), int(h * b))


def fit_pil(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / image.width, max_h / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (max_w, max_h), "white")
    canvas.paste(resized, ((max_w - resized.width) // 2, (max_h - resized.height) // 2))
    return canvas


def save_review_visuals(record: dict[str, Any], output_path: Path, visuals: dict[str, np.ndarray] | None = None) -> None:
    request_id = record["request_id"]
    source_path = Path(record["source_condition_path"])
    side = ATTEMPT_ROOT / "05_review_assets" / "side_by_side" / f"{request_id}_source_output_side_by_side.png"
    full = ATTEMPT_ROOT / "05_review_assets" / "full_body" / f"{request_id}_full_body_page.png"
    side.parent.mkdir(parents=True, exist_ok=True)
    full.parent.mkdir(parents=True, exist_ok=True)
    pair_canvas(source_path, output_path, request_id).save(side, format="PNG")
    pair_canvas(source_path, output_path, f"{request_id} full_body").save(full, format="PNG")
    crop_defs = {
        "face_head": (0.32, 0.00, 0.68, 0.28),
        "neck_shoulder": (0.23, 0.10, 0.77, 0.40),
        "hands_feet": (0.08, 0.34, 0.92, 1.00),
        "garment_boundary": (0.16, 0.14, 0.84, 0.90),
    }
    crop_paths = {}
    for name, box in crop_defs.items():
        path = ATTEMPT_ROOT / "05_review_assets" / "crops" / request_id / f"{name}_source_output_crop.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        pair_canvas(source_path, output_path, f"{request_id} {name}", box).save(path, format="PNG")
        crop_paths[name] = str(path)
    record["review_assets"] = {
        "side_by_side": str(side),
        "full_body": str(full),
        "crops": crop_paths,
        "layout_label": "DISPLAY_ONLY_REVIEW_LAYOUT",
    }
    if visuals:
        visual_dir = ATTEMPT_ROOT / "05_review_assets" / "registration_visuals" / request_id
        visual_dir.mkdir(parents=True, exist_ok=True)
        module = load_registration_module()
        visual_paths = {}
        for name, image in visuals.items():
            if image is None:
                continue
            path = visual_dir / f"{name}.png"
            module.cv2.imwrite(str(path), image)
            visual_paths[name] = str(path)
        record["review_assets"]["registration_visuals"] = visual_paths


def build_contact_sheet(records: list[dict[str, Any]]) -> str | None:
    raw = [record for record in records if record.get("raw_output_path")]
    if not raw:
        return None
    tile_w, tile_h, label_h = 520, 448, 48
    cols = 2
    rows = (len(raw) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, record in enumerate(raw):
        with Image.open(record["raw_output_path"]) as image:
            tile = fit_pil(image.convert("RGB"), tile_w, tile_h)
        x = (index % cols) * tile_w
        y = (index // cols) * (tile_h + label_h)
        canvas.paste(tile, (x, y))
        draw.text((x + 8, y + tile_h + 6), record["request_id"], fill="black", font=font)
        draw.text((x + 8, y + tile_h + 24), f"{record['raw_output_resolution']} | DISPLAY_ONLY_REVIEW_LAYOUT", fill="black", font=font)
    path = ATTEMPT_ROOT / "05_review_assets" / "contact_sheet" / "subject00_o03_canary_correction_continuation_contact_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    return str(path)


def review_manifest_for_state(state: dict[str, Any], contact_sheet: str | None) -> dict[str, Any]:
    review_records = []
    for record in state["records"]:
        if not record.get("raw_output_path"):
            continue
        review_records.append(
            {
                "request_id": record["request_id"],
                "garment": record["garment"],
                "slot": record["slot"],
                "camera": record["camera"],
                "direction": record["direction"],
                "source_condition_path": record["source_condition_path"],
                "source_condition_sha256": record["source_condition_sha256"],
                "raw_output_path": record["raw_output_path"],
                "raw_output_sha256": record["raw_output_sha256"],
                "raw_output_resolution": record["raw_output_resolution"],
                "technical_status": record["corrected_technical_status"],
                "machine_registration_status": record.get("machine_registration_status"),
                "machine_registration_primary_classification": record.get("machine_registration_primary_classification"),
                "review_assets": record.get("review_assets", {}),
                "human_visual_decision": None,
                "hood_removal_pass": None,
                "identity_pass": None,
                "camera_pose_pass": None,
                "background_pass": None,
                "garment_pass": None,
                "accepted": False,
                "teacher_target": False,
            }
        )
    return {
        "schema_version": "canondressgs.subject00.o03_canary.correction_continuation_human_review_manifest.v1",
        "task_id": TASK_ID,
        "attempt_namespace": ATTEMPT_NAMESPACE,
        "status": "PENDING_USER_REVIEW",
        "contact_sheet_path": contact_sheet,
        "record_count": len(review_records),
        "human_visual_decision": None,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "records": review_records,
        "paper_final": False,
    }


def finalize() -> dict[str, Any]:
    state = load_state()
    update_state_counts(state)
    contact_sheet = build_contact_sheet(state["records"])
    comparison = old_attempt_comparison()
    if any(item["mutations"] for item in comparison.values()):
        final_classification = "SUBJECT00_O03_CANARY_CONTRACT_VIOLATION_OLD_ATTEMPT_MUTATION"
        next_task = "USER_RESOLVE_SUBJECT00_O03_CANARY_OLD_ATTEMPT_MUTATION"
    elif state["raw_generated_output_count"] == 4 and state["resolution_family_pass_count"] == 4 and state["machine_registration_pass_count"] == 4 and contact_sheet:
        final_classification = "SUBJECT00_O03_CANARY_TECHNICAL_PASS_PENDING_HUMAN_REVIEW"
        next_task = "USER_REVIEW_SUBJECT00_O03_CANARY_OUTPUTS"
    elif state["raw_generated_output_count"] == 4 and contact_sheet:
        final_classification = "SUBJECT00_O03_CANARY_4_REQUESTS_COMPLETED_PENDING_HUMAN_REVIEW"
        next_task = "USER_REVIEW_SUBJECT00_O03_CANARY_OUTPUTS"
    else:
        final_classification = "SUBJECT00_O03_CANARY_PARTIAL_PENDING_HUMAN_REVIEW"
        next_task = "USER_REVIEW_SUBJECT00_O03_CANARY_OUTPUTS"

    state["status"] = final_classification
    state["finalized_at"] = now()
    state["old_attempt_comparison_after"] = comparison
    state["contact_sheet_path"] = contact_sheet
    save_state(state)

    raw_records = [record for record in state["records"] if record.get("raw_output_path")]
    raw_registry = {
        "schema_version": "canondressgs.subject00.o03_canary.corrected_raw_output_registry.v1",
        "task_id": TASK_ID,
        "attempt_namespace": ATTEMPT_NAMESPACE,
        "raw_generated_output_count": state["raw_generated_output_count"],
        "png_parse_pass_count": state["png_parse_pass_count"],
        "resolution_family_pass_count": state["resolution_family_pass_count"],
        "machine_registration_pass_count": state["machine_registration_pass_count"],
        "human_review_eligible_count": state["human_review_eligible_count"],
        "records": [
            {
                key: record.get(key)
                for key in [
                    "request_id",
                    "slot",
                    "camera",
                    "direction",
                    "source_condition_path",
                    "source_condition_sha256",
                    "raw_output_path",
                    "raw_output_sha256",
                    "raw_output_resolution",
                    "png_parse_status",
                    "resolution_family_pass",
                    "sha_unique",
                    "source_binding_status",
                    "corrected_technical_status",
                    "machine_registration_status",
                    "machine_registration_primary_classification",
                    "human_review_eligible",
                ]
            }
            for record in raw_records
        ],
        "paper_final": False,
    }
    review_manifest = review_manifest_for_state(state, contact_sheet)
    summary = {
        "schema_version": "canondressgs.subject00.o03_canary.resolution_correction_final_summary.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "new_worktree": str(ROOT),
        "original_contract_path": str(ROOT / FROZEN_CONTRACT_REL),
        "correction_overlay_path": str(ROOT / OVERLAY_REL),
        "correction_report_path": str(ROOT / CORRECTION_REPORT_REL),
        "attempt_namespace": ATTEMPT_NAMESPACE,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "total_authorized_generation_calls": 4,
        "generation_calls_before_task": 1,
        "additional_generation_calls": state["additional_generation_calls"],
        "total_generation_calls": state["total_generation_calls"],
        "retry_calls": 0,
        "request_ids": EXPECTED_REQUEST_IDS,
        "source_condition_paths": [record["source_condition_path"] for record in state["records"]],
        "source_condition_sha256": [record["source_condition_sha256"] for record in state["records"]],
        "raw_native_resolution_accepted_set": state["raw_native_resolution_accepted_set"],
        "raw_generated_output_count": state["raw_generated_output_count"],
        "raw_output_paths": state["raw_output_paths"],
        "raw_output_resolutions": state["raw_output_resolutions"],
        "raw_output_sha256": state["raw_output_sha256"],
        "png_parse_pass_count": state["png_parse_pass_count"],
        "resolution_family_pass_count": state["resolution_family_pass_count"],
        "machine_registration_pass_count": state["machine_registration_pass_count"],
        "human_review_eligible_count": state["human_review_eligible_count"],
        "first_output_original_status": "NATIVE_RESOLUTION_MISMATCH_UNDER_OLD_EXACT_GATE",
        "first_output_corrected_status": state["records"][0].get("corrected_technical_status"),
        "first_output_technical_admissibility": state["records"][0].get("first_output_technical_admissibility"),
        "postprocessing_calls": 0,
        "old_attempt_comparison": comparison,
        "contact_sheet_path": contact_sheet,
        "side_by_side_review_paths": [
            record.get("review_assets", {}).get("side_by_side") for record in raw_records if record.get("review_assets")
        ],
        "crop_review_paths": [
            path
            for record in raw_records
            for path in record.get("review_assets", {}).get("crops", {}).values()
        ],
        "review_manifest_path": str(ROOT / REVIEW_MANIFEST_REL),
        "human_visual_decision": None,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "paper_modifications": 0,
        "paper_final": False,
        "final_classification": final_classification,
        "next_task": next_task,
    }

    write_json(RAW_REGISTRY_EXTERNAL_PATH, raw_registry)
    write_json(REVIEW_MANIFEST_EXTERNAL_PATH, review_manifest)
    write_json(FINAL_SUMMARY_EXTERNAL_PATH, summary)
    write_json(ROOT / CONTINUATION_MANIFEST_REL, state)
    write_json(ROOT / RAW_REGISTRY_REL, raw_registry)
    write_json(ROOT / REVIEW_MANIFEST_REL, review_manifest)
    write_json(ROOT / FINAL_SUMMARY_REL, summary)
    write_json(ROOT / HANDOFF_REL, {"task_id": TASK_ID, "execution_status": final_classification, "summary": str(ROOT / FINAL_SUMMARY_REL), "next_task": next_task})
    write_report(summary)
    tests = validate_final(write=True)
    summary["test_result"] = tests["status"]
    write_json(ROOT / FINAL_SUMMARY_REL, summary)
    write_json(FINAL_SUMMARY_EXTERNAL_PATH, summary)
    return summary


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Subject00 O03 Canary Resolution Correction Continuation Report",
        "",
        f"- Task ID: `{TASK_ID}`",
        f"- Source: `{SOURCE_BRANCH}` @ `{SOURCE_HEAD}`",
        f"- Branch: `{NEW_BRANCH}`",
        f"- Attempt namespace: `{ATTEMPT_NAMESPACE}`",
        f"- Corrected resolution family: `{ACCEPTED_FAMILY_NAME}`",
        f"- Raw generated output count: `{summary['raw_generated_output_count']}`",
        f"- PNG parse pass count: `{summary['png_parse_pass_count']}`",
        f"- Resolution family pass count: `{summary['resolution_family_pass_count']}`",
        f"- Machine registration pass count: `{summary['machine_registration_pass_count']}`",
        f"- Human-review eligible count: `{summary['human_review_eligible_count']}`",
        f"- Additional generation calls: `{summary['additional_generation_calls']}`",
        f"- Retry calls: `0`",
        f"- Postprocessing calls: `0`",
        f"- Contact sheet: `{summary['contact_sheet_path']}`",
        f"- Final classification: `{summary['final_classification']}`",
        f"- Next task: `{summary['next_task']}`",
        "",
        "All human visual fields remain null. Accepted and Teacher-target counts remain zero.",
    ]
    write_text(ROOT / CORRECTION_REPORT_REL, "\n".join(lines))


def validate_final(write: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str | None = None) -> None:
        item = {"name": name, "status": "PASS" if passed else "FAIL"}
        if detail:
            item["detail"] = detail
        checks.append(item)

    try:
        state = load_state()
        summary = read_json(ROOT / FINAL_SUMMARY_REL) if (ROOT / FINAL_SUMMARY_REL).exists() else {}
        overlay = read_json(ROOT / OVERLAY_REL)
        review = read_json(ROOT / REVIEW_MANIFEST_REL) if (ROOT / REVIEW_MANIFEST_REL).exists() else {}
        check("task_id", state["task_id"] == TASK_ID and overlay["task_id"] == TASK_ID)
        check("source_branch_head", state["source_branch"] == SOURCE_BRANCH and state["source_head"] == SOURCE_HEAD)
        check("new_branch", git(ROOT, "branch", "--show-current") == NEW_BRANCH)
        check("original_contract_unmodified", FROZEN_CONTRACT_REL.as_posix() not in git(ROOT, "diff", "--name-only", SOURCE_HEAD, "--").splitlines())
        check("original_manifest_unmodified", FROZEN_MANIFEST_REL.as_posix() not in git(ROOT, "diff", "--name-only", SOURCE_HEAD, "--").splitlines())
        check("generation_budget", state["additional_generation_calls"] <= 3 and state["total_generation_calls"] <= 4)
        check("retry_postprocess_zero", state["retry_calls"] == 0 and state["postprocessing_calls"] == 0)
        check("first_raw_preserved", FIRST_RAW_PATH.is_file() and file_sha256(FIRST_RAW_PATH) == FIRST_EXPECTED_SHA)
        comparison = old_attempt_comparison()
        check("attempt_001_mutations_zero", comparison["attempt_001"]["mutations"] == 0)
        check("attempt_002_mutations_zero", comparison["attempt_002"]["mutations"] == 0)
        check("attempt_003_mutations_zero", comparison["attempt_003"]["mutations"] == 0)
        raw_records = [record for record in state["records"] if record.get("raw_output_path")]
        check("raw_counts_consistent", state["raw_generated_output_count"] == len(raw_records))
        check("png_parse_counts_consistent", state["png_parse_pass_count"] == sum(record.get("png_parse_status") == "PASS" for record in raw_records))
        check("resolution_family_counts_consistent", state["resolution_family_pass_count"] == sum(bool(record.get("resolution_family_pass")) for record in raw_records))
        check("machine_registration_counts_consistent", state["machine_registration_pass_count"] == sum(record.get("machine_registration_status") == "PASS" for record in raw_records))
        check("human_fields_null", review.get("human_visual_decision") is None and all(item.get("human_visual_decision") is None for item in review.get("records", [])))
        check("no_accept_teacher", review.get("accepted_count", 0) == 0 and review.get("teacher_target_count", 0) == 0)
        check("paper_flags", summary.get("paper_modifications", 0) == 0 and summary.get("paper_final") is False)
        check("registration_contract_recovered", state["registration_contract"]["status"] == "RECOVERED_UNIQUELY")
        check("review_pack_formed_for_raw", all(record.get("review_assets") for record in raw_records))
    except Exception as exc:  # noqa: BLE001
        check("validate_final_exception", False, f"{type(exc).__name__}: {exc}")

    result = {
        "schema_version": "canondressgs.subject00.o03_canary.resolution_correction_tests.v1",
        "task_id": TASK_ID,
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "test_count": len(checks),
        "fail_count": sum(item["status"] == "FAIL" for item in checks),
        "checks": checks,
        "paper_final": False,
    }
    if write:
        write_json(ROOT / TESTS_REL, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("preflight", "write-correction-overlay", "check-first", "record-output", "finalize", "validate-final"),
    )
    parser.add_argument("--request-id")
    parser.add_argument("--tool-output", type=Path)
    parser.add_argument("--tool-output-hint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "preflight":
        result = validate_preflight()
    elif args.action == "write-correction-overlay":
        result = write_correction_overlay()
    elif args.action == "check-first":
        result = check_first()
    elif args.action == "record-output":
        if not args.request_id or not args.tool_output:
            raise ValueError("record-output requires --request-id and --tool-output")
        result = record_generated_output(args.request_id, args.tool_output.resolve(), args.tool_output_hint)
    elif args.action == "finalize":
        result = finalize()
    else:
        result = validate_final(write=False)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if not isinstance(result, dict) or result.get("status") != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
