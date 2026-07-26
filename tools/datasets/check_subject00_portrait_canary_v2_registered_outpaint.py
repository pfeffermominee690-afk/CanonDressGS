#!/usr/bin/env python3
"""Independently verify the Subject00 V2 registered-outpaint canary contract."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
SOURCE_WORKTREE = Path(r"E:\model_train\canondressgs_subject00_portrait_canary_visual_fail_adjudication")
ATTEMPT_001 = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_001")
ATTEMPT_002 = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_002_portrait_canary")
ATTEMPT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_003_portrait_canary_v2_registered_outpaint")

TASK_ID = "AAAI27-SUBJECT00-PORTRAIT-CANARY-V2-REGISTERED-OUTPAINT-001"
SOURCE_BRANCH = "research/subject00-portrait-canary-visual-fail-adjudication-20260726"
SOURCE_HEAD = "5a992921a5e33d1a23a332094ee9383f441217bd"
BRANCH = "research/subject00-portrait-canary-v2-registered-outpaint-20260726"
CLASSIFICATION = "SUBJECT00_PORTRAIT_CANARY_V2_TECHNICAL_PASS_PENDING_USER_REVIEW"
NEXT_TASK = "USER_REVIEW_SUBJECT00_PORTRAIT_CANARY_V2_CONTACT_SHEET"
REQUEST_IDS = [
    "subject00_O01_slot00_cand00",
    "subject00_O03_slot03_cand00",
    "subject00_O04_slot02_cand00",
    "subject00_O01_slot06_cand00",
]

PREFLIGHT_PATH = RISK / "subject00_portrait_canary_v2_registration_preflight.json"
OUTPUTS_PATH = RISK / "subject00_portrait_canary_v2_output_registry.json"
REVIEW_CONTRACT_PATH = RISK / "subject00_portrait_canary_v2_review_contract.json"
SUMMARY_PATH = RISK / "subject00_portrait_canary_v2_final_summary.json"
TESTS_PATH = RISK / "subject00_portrait_canary_v2_tests.json"
HANDOFF_PATH = ROOT / "project_control_handoff" / "subject00_portrait_canary_v2_registered_outpaint_handoff.json"
EXTERNAL_TESTS_PATH = ATTEMPT / "09_final_verification" / "subject00_portrait_canary_v2_tests.json"
BASELINE_PATH = ATTEMPT / "00_source_evidence" / "prior_attempt_immutability_baseline.json"
PROGRESS_PATH = ATTEMPT / "05_generation_requests" / "generation_progress.json"
REVIEW_MANIFEST_PATH = ATTEMPT / "07_human_review" / "subject00_portrait_canary_v2_review_manifest.json"
OVERLAY_PATH = ATTEMPT / "01_registration_preflight" / "registration_overlay_contact_sheet.png"
CONTACT_SHEET_PATH = ATTEMPT / "07_human_review" / "subject00_portrait_canary_v2_contact_sheet.png"


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
    payload = dict(payload)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run(*command: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        list(command), cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def assert_true(value: Any, message: str) -> None:
    if not value:
        raise AssertionError(message)


def inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, file_sha256(path))
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def baseline_inventory(records: list[dict[str, Any]]) -> dict[str, tuple[int, str]]:
    return {item["relative_path"]: (item["bytes"], item["sha256"]) for item in records}


def image_equal(left: Image.Image, right: Image.Image) -> bool:
    return left.mode == right.mode and left.size == right.size and ImageChops.difference(left, right).getbbox() is None


def main() -> int:
    preflight = load_json(PREFLIGHT_PATH)
    outputs = load_json(OUTPUTS_PATH)
    review_contract = load_json(REVIEW_CONTRACT_PATH)
    summary = load_json(SUMMARY_PATH)
    handoff = load_json(HANDOFF_PATH)
    baseline = load_json(BASELINE_PATH)
    progress = load_json(PROGRESS_PATH)
    review = load_json(REVIEW_MANIFEST_PATH)
    records = preflight["records"]
    output_records = outputs["records"]
    output_by_id = {item["request_id"]: item for item in output_records}
    checks: list[dict[str, Any]] = []

    def check(name: str, function: Callable[[], None]) -> None:
        try:
            function()
            checks.append({"name": name, "status": "PASS"})
        except Exception as error:  # noqa: BLE001 - every failed gate must be retained
            checks.append({"name": name, "status": "FAIL", "detail": f"{type(error).__name__}: {error}"})

    check("source_branch", lambda: assert_true(
        run("git", "branch", "--show-current", cwd=SOURCE_WORKTREE) == SOURCE_BRANCH,
        "source branch mismatch",
    ))
    check("source_head", lambda: assert_true(
        run("git", "rev-parse", "HEAD", cwd=SOURCE_WORKTREE) == SOURCE_HEAD,
        "source HEAD mismatch",
    ))
    check("source_worktree_clean", lambda: assert_true(
        run("git", "status", "--short", cwd=SOURCE_WORKTREE) == "",
        "source worktree is dirty",
    ))
    check("target_branch", lambda: assert_true(run("git", "branch", "--show-current") == BRANCH, "wrong target branch"))
    check("source_head_ancestor", lambda: assert_true(
        subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT).returncode == 0,
        "source HEAD is not an ancestor",
    ))
    check("attempt_001_immutability", lambda: assert_true(
        inventory(ATTEMPT_001) == baseline_inventory(baseline["attempt_001"]),
        "attempt_001 inventory changed",
    ))
    check("attempt_002_immutability", lambda: assert_true(
        inventory(ATTEMPT_002) == baseline_inventory(baseline["attempt_002"]),
        "attempt_002 inventory changed",
    ))
    check("calibration_source_sealed", lambda: assert_true(
        file_sha256(ATTEMPT / "00_source_evidence" / "calibration.json")
        == preflight["calibration_sha256"]
        == "4b2d98d20740da03be209040851fab7e8801e5670937ed9ad290a20264d1dde7",
        "calibration SHA mismatch",
    ))
    check("camera_binding_completeness", lambda: assert_true(
        preflight["camera_binding_status"] == "PASS"
        and len(records) == 4
        and all({"camera", "identity_condition_mask", "identity_condition_rgb", "pose_smplx"}.issubset(item["camera_binding"]) for item in records),
        "camera binding incomplete",
    ))
    check("camera_intrinsics_provenance", lambda: assert_true(
        preflight["camera_intrinsics_status"] == "PASS"
        and all(set(item["camera_intrinsics"]) == {"fx", "fy", "cx", "cy"} for item in records)
        and all(len(item["extrinsics"]["world_to_camera_rotation_R"]) == 9 for item in records)
        and all(len(item["extrinsics"]["world_to_camera_translation_T"]) == 3 for item in records),
        "intrinsics or extrinsics provenance incomplete",
    ))
    check("four_request_completeness", lambda: assert_true(
        [item["request_id"] for item in records] == REQUEST_IDS
        and progress["stable_order"] == REQUEST_IDS
        and [item["request_id"] for item in output_records] == REQUEST_IDS,
        "request set or order mismatch",
    ))
    check("crop_window_bounds", lambda: assert_true(
        all(0 <= item["crop_window"]["x_left"] <= 306
            and item["crop_window"]["x_right"] - item["crop_window"]["x_left"] == 1024
            and item["crop_window"]["y_top"] == 0
            and item["crop_window"]["y_bottom"] == 1150 for item in records),
        "crop window outside source bounds",
    ))
    check("principal_point_crop_rule", lambda: assert_true(
        all(item["crop_window"]["x_left"] == max(0, min(round(item["camera_intrinsics"]["cx"] - 512), 306)) for item in records),
        "crop is not principal-point registered",
    ))
    check("crop_person_intersection", lambda: assert_true(
        all(item["crop_person_intersection_pixels"] == 0 for item in records),
        "crop intersects person mask",
    ))
    check("crop_garment_intersection", lambda: assert_true(
        all(item["crop_garment_intersection_pixels"] == 0 for item in records),
        "crop intersects garment envelope",
    ))
    check("crop_hands_feet_intersection", lambda: assert_true(
        all(item["crop_hands_feet_intersection_pixels"] == 0 for item in records),
        "crop intersects hands/feet envelope",
    ))
    check("no_source_scaling", lambda: assert_true(
        all(item["source_region_scale_x"] == item["source_region_scale_y"] == 1.0 for item in records),
        "source scaling detected",
    ))
    check("updated_intrinsics", lambda: assert_true(
        all(item["updated_intrinsics"]["fx"] == item["camera_intrinsics"]["fx"]
            and item["updated_intrinsics"]["fy"] == item["camera_intrinsics"]["fy"]
            and item["updated_intrinsics"]["cx"] == item["camera_intrinsics"]["cx"] - item["crop_window"]["x_left"]
            and item["updated_intrinsics"]["cy"] == item["camera_intrinsics"]["cy"] + 193 for item in records),
        "updated intrinsics mismatch",
    ))
    check("vertical_extension_mapping", lambda: assert_true(
        all(item["top_extension"] == item["bottom_extension"] == 193 for item in records),
        "vertical extension mismatch",
    ))
    check("central_destination_contract", lambda: assert_true(
        all(item["source_region_destination"] == {"x_left": 0, "x_right": 1024, "y_top": 193, "y_bottom": 1343} for item in records),
        "central destination mismatch",
    ))

    def verify_registered_pixels() -> None:
        for item in records:
            with Image.open(item["source_path"]) as source_image, Image.open(item["registered_crop_path"]) as crop_image:
                source = source_image.convert("RGB")
                crop = crop_image.convert("RGB")
                x_left = item["crop_window"]["x_left"]
                assert_true(image_equal(source.crop((x_left, 0, x_left + 1024, 1150)), crop), item["request_id"])

    check("registered_crop_pixel_identity", verify_registered_pixels)

    def verify_canvas_pixels() -> None:
        for item in records:
            with Image.open(item["registered_crop_path"]) as crop_image, Image.open(item["derived_canvas_path"]) as canvas_image:
                crop = crop_image.convert("RGB")
                canvas = canvas_image.convert("RGB")
                assert_true(image_equal(crop, canvas.crop((0, 193, 1024, 1343))), item["request_id"])

    check("derived_central_pixel_identity", verify_canvas_pixels)

    def verify_canvas_resolution() -> None:
        for item in records:
            with Image.open(item["derived_canvas_path"]) as image:
                image.load()
                assert_true(image.format == "PNG" and image.size == (1024, 1536), item["request_id"])

    check("derived_canvas_resolution", verify_canvas_resolution)

    def verify_edit_maps() -> None:
        for item in records:
            with Image.open(item["edit_mask_path"]) as map_image, Image.open(item["source_mask_path"]) as mask_image:
                edit_map = map_image.convert("RGB")
                source_mask = mask_image.convert("L").point(lambda value: 255 if value >= 128 else 0)
                x_left = item["crop_window"]["x_left"]
                crop_mask = source_mask.crop((x_left, 0, x_left + 1024, 1150))
                expected = Image.new("RGB", (1024, 1536), (0, 0, 0))
                expected.paste((255, 0, 0), (0, 0, 1024, 193))
                expected.paste((0, 0, 255), (0, 1343, 1024, 1536))
                expected.paste(Image.new("RGB", (1024, 1150), (0, 255, 0)), (0, 193), crop_mask)
                assert_true(image_equal(edit_map, expected), item["request_id"])

    check("edit_region_map_exactness", verify_edit_maps)
    check("registration_overlay_completeness", lambda: assert_true(
        OVERLAY_PATH.is_file() and file_sha256(OVERLAY_PATH)
        == file_sha256(ATTEMPT / "01_registration_preflight" / "registration_overlay_contact_sheet.png"),
        "registration overlay missing",
    ))
    check("phase_0_gate", lambda: assert_true(
        preflight["phase_0_status"] == progress["phase_0_status"] == summary["phase_0_status"] == "REGISTRATION_PREFLIGHT_PASS"
        and all(item["registration_decision"] == "REGISTRATION_PREFLIGHT_PASS" for item in records),
        "Phase 0 did not pass 4/4",
    ))
    check("generation_call_count", lambda: assert_true(
        progress["generation_calls"] == outputs["generation_calls"] == summary["generation_calls"] == 4,
        "generation call count mismatch",
    ))
    check("one_call_per_request", lambda: assert_true(
        len(progress["records"]) == 4
        and len({item["generation_call_id"] for item in progress["records"]}) == 4
        and {item["request_id"] for item in progress["records"]} == set(REQUEST_IDS),
        "one-call-per-request contract failed",
    ))
    check("no_silent_retry", lambda: assert_true(
        progress["technical_retries"] == outputs["technical_retries"] == summary["technical_retries"] == 0
        and all(item["retry_count"] == 0 for item in output_records),
        "retry detected",
    ))
    check("single_output_per_request", lambda: assert_true(
        len(list((ATTEMPT / "06_generation_responses").rglob("*.png"))) == 4
        and all(Path(item["output_path"]).parent.name == item["request_id"] for item in output_records),
        "output count or isolation mismatch",
    ))

    def verify_native_outputs() -> None:
        for item in output_records:
            with Image.open(item["output_path"]) as image:
                image.load()
                assert_true(image.size == (1024, 1536), item["request_id"])

    check("native_output_resolution", verify_native_outputs)

    def verify_png_parse() -> None:
        for item in output_records:
            with Image.open(item["output_path"]) as image:
                image.verify()
                assert_true(image.format == "PNG", item["request_id"])

    check("png_parse", verify_png_parse)
    check("output_sha_provenance", lambda: [
        assert_true(file_sha256(Path(item["output_path"])) == item["output_sha256"], item["request_id"])
        for item in output_records
    ])
    check("managed_provenance_complete", lambda: [
        assert_true(
            (lambda provenance: provenance["generation_backend"] == "CODEX_MANAGED_IMAGE_EDIT"
             and provenance["generation_call_id"] == item["generation_call_id"]
             and provenance["output_sha256"] == item["output_sha256"]
             and provenance["postprocessing_count"] == provenance["retry_count"] == 0)(
                load_json(ATTEMPT / "08_provenance" / "requests" / f"{item['request_id']}.json")
            ),
            item["request_id"],
        ) for item in output_records
    ])
    check("duplicate_sha_zero", lambda: assert_true(
        outputs["duplicate_sha_count"] == 0 and len({item["output_sha256"] for item in output_records}) == 4,
        "duplicate output SHA detected",
    ))
    check("no_output_postprocessing", lambda: assert_true(
        outputs["postprocessing_count"] == summary["postprocessing_count"] == 0
        and all(item["postprocessing_count"] == 0 for item in output_records),
        "output postprocessing detected",
    ))

    def verify_nonconstant_extension_bands() -> None:
        for item in output_records:
            with Image.open(item["output_path"]) as image:
                rgb = image.convert("RGB")
                for box in [(0, 0, 1024, 193), (0, 1343, 1024, 1536)]:
                    colors = rgb.crop(box).resize((128, 24)).getcolors(maxcolors=128 * 24)
                    assert_true(colors is not None and len(colors) > 64, item["request_id"])

    check("no_constant_padding_bands", verify_nonconstant_extension_bands)
    check("no_external_api", lambda: assert_true(
        progress["external_api_calls"] == summary["external_api_calls"] == 0,
        "external API call recorded",
    ))
    check("no_api_key_read", lambda: assert_true(
        progress["api_key_reads"] == summary["api_key_reads"] == 0,
        "API key read recorded",
    ))
    check("no_cloud_image_write", lambda: assert_true(
        progress["cloud_image_writes"] == summary["cloud_image_writes"] == 0,
        "cloud image write recorded",
    ))

    def changed_repo_paths() -> list[str]:
        tracked = run("git", "diff", "--name-only", SOURCE_HEAD, "--").splitlines()
        untracked = run("git", "ls-files", "--others", "--exclude-standard").splitlines()
        return sorted(set(tracked + untracked))

    check("no_image_git_commit", lambda: assert_true(
        not any(Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"} for path in changed_repo_paths()),
        "image material entered Git changes",
    ))
    check("accepted_count_zero", lambda: assert_true(
        summary["accepted_count"] == handoff["accepted_count"] == review["accepted_count"] == 0,
        "accepted count is nonzero",
    ))
    check("teacher_target_count_zero", lambda: assert_true(
        summary["teacher_target_count"] == handoff["teacher_target_count"] == review["teacher_target_count"] == 0,
        "Teacher target count is nonzero",
    ))
    check("remaining_39_zero", lambda: assert_true(
        summary["remaining_39_generated_count"] == 0
        and not review_contract["remaining_39_rerun_authorized"]
        and not handoff["remaining_39_rerun_authorized"],
        "remaining 39 were generated or authorized",
    ))
    check("formal_base_pending", lambda: assert_true(
        summary["formal_base"] == handoff["formal_base"] == "PENDING",
        "Formal Base status changed",
    ))
    check("no_paper_modification", lambda: assert_true(
        summary["paper_modifications"] == 0
        and not any(Path(path).suffix.lower() == ".tex" for path in changed_repo_paths()),
        "paper source changed",
    ))
    check("human_review_fields_null", lambda: assert_true(
        review["visual_decision_count"] == 0
        and len(review["records"]) == 4
        and all(item["visual_review_status"] == "PENDING_USER_REVIEW" for item in review["records"])
        and all(all(item[field] is None for field in [
            "identity_match", "pose_match", "camera_match", "garment_match", "full_body_complete",
            "hands_complete", "feet_complete", "background_match", "artifact_grade", "decision",
        ]) for item in review["records"]),
        "human visual field was populated",
    ))
    check("contact_sheet_readable", lambda: (
        (lambda image: (image.load(), assert_true(image.format == "PNG" and image.width > 0 and image.height > 0, "contact sheet invalid")))(Image.open(CONTACT_SHEET_PATH))
    ))
    check("metrics_explicitly_unavailable", lambda: assert_true(
        all(item["subject_height_ratio"] == item["subject_center_displacement"] == item["keypoint_drift"] == "NOT_AVAILABLE" for item in output_records),
        "unfrozen metric was inferred",
    ))
    check("no_acceptance_directories", lambda: assert_true(
        not any(path.is_dir() and ("accepted" in path.name.lower() or "teacher" in path.name.lower()) for path in ATTEMPT.rglob("*")),
        "accepted or Teacher directory exists",
    ))
    check("final_classification", lambda: assert_true(
        summary["classification"] == handoff["classification"] == CLASSIFICATION,
        "final classification mismatch",
    ))
    check("next_task_unique", lambda: assert_true(
        {summary["next_task"], handoff["next_task"]} == {NEXT_TASK}
        and not summary["next_task_authorized"]
        and not handoff["next_task_authorized"],
        "NEXT_TASK mismatch or authorization leak",
    ))
    check("paper_final_false", lambda: assert_true(
        all(payload["paper_final"] is False for payload in [outputs, review_contract, summary, handoff]),
        "PAPER_FINAL is true",
    ))
    check("content_sha256_seals", lambda: [
        assert_true(canonical_sha256(payload) == payload["content_sha256"], path.name)
        for path, payload in [
            (PREFLIGHT_PATH, preflight), (OUTPUTS_PATH, outputs), (REVIEW_CONTRACT_PATH, review_contract),
            (SUMMARY_PATH, summary), (HANDOFF_PATH, handoff), (BASELINE_PATH, baseline),
            (PROGRESS_PATH, progress), (REVIEW_MANIFEST_PATH, review),
        ]
    ])
    check("no_secret_like_values", lambda: assert_true(
        re.search(
            r"(?i)(sk-[a-z0-9_-]{16,}|bearer\s+[a-z0-9._-]{16,}|api[_-]?key\s*[:=]\s*[\"'][^\"']{8,})",
            "\n".join(path.read_text(encoding="utf-8") for path in [PREFLIGHT_PATH, OUTPUTS_PATH, SUMMARY_PATH, HANDOFF_PATH]),
        ) is None,
        "secret-like value found",
    ))

    failed = [item for item in checks if item["status"] == "FAIL"]
    status = f"PASS_{len(checks)}_REGISTERED_OUTPAINT_CHECKS" if not failed else f"FAIL_{len(failed)}_OF_{len(checks)}"
    tests = {
        "schema_version": "canondressgs.subject00.portrait_canary_v2_tests.v1",
        "task_id": TASK_ID,
        "status": status,
        "total": len(checks),
        "passed": len(checks) - len(failed),
        "failed": failed,
        "checks": checks,
        "paper_final": False,
    }
    write_json(TESTS_PATH, tests)
    write_json(EXTERNAL_TESTS_PATH, tests)
    for path in [SUMMARY_PATH, HANDOFF_PATH, ATTEMPT / "09_final_verification" / "subject00_portrait_canary_v2_final_summary.json"]:
        payload = load_json(path)
        payload["tests"] = status
        write_json(path, payload)
    print(json.dumps({"status": status, "total": len(checks), "failed": failed}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
