#!/usr/bin/env python3
"""Independently verify the Subject00 Attempt 001 native-landscape audit."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
SOURCE_WORKTREE = Path(r"E:\model_train\canondressgs_subject00_v2_visual_fail_valid_region_audit")
ATTEMPTS_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001")
ATTEMPTS = {
    "attempt_001": ATTEMPTS_ROOT / "attempt_001",
    "attempt_002": ATTEMPTS_ROOT / "attempt_002_portrait_canary",
    "attempt_003": ATTEMPTS_ROOT / "attempt_003_portrait_canary_v2_registered_outpaint",
}
AUDIT_ROOT = ATTEMPTS_ROOT / "attempt_001_native_landscape_registration_audit"
ARCHIVE = Path(r"E:\data_pre\thuman4_second_identity_staging\downloads\subject00.7z")
CLOUD_MANIFEST = Path(r"E:\data_pre\thuman4_second_identity_staging\reports\SUBJECT00_CLOUD_DATA_MANIFEST.json")
CALIBRATION = ATTEMPTS["attempt_003"] / "00_source_evidence" / "calibration.json"

TASK_ID = "AAAI27-SUBJECT00-ATTEMPT001-NATIVE-LANDSCAPE-REGISTRATION-AUDIT-001"
SOURCE_BRANCH = "research/subject00-v2-visual-fail-valid-region-audit-20260726"
SOURCE_HEAD = "34231cd3a203732b426f4f88701f22908cd26a56"
BRANCH = "research/subject00-attempt001-native-landscape-registration-audit-20260726"
CALIBRATION_SHA256 = "4b2d98d20740da03be209040851fab7e8801e5670937ed9ad290a20264d1dde7"
FINAL_CLASSIFICATION = "SUBJECT00_NATIVE_LANDSCAPE_PARTIAL_SALVAGE_TARGETED_GAPS_IDENTIFIED"
NEXT_TASK = "USER_REVIEW_SALVAGE_CANDIDATES_AND_AUTHORIZE_ONLY_MISSING_CELL_CANARY"

GARMENTS = ("O01", "O03", "O04")
SLOT_CAMERAS = {
    "slot_00": "cam17", "slot_01": "cam21", "slot_02": "cam14", "slot_03": "cam23",
    "slot_04": "cam11", "slot_05": "cam02", "slot_06": "cam09", "slot_07": "cam05",
}
CLASSIFICATIONS = (
    "REGISTERED_SIMILARITY_PASS_CANDIDATE",
    "REGISTRATION_EVIDENCE_INSUFFICIENT",
    "AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL",
    "BACKGROUND_GEOMETRY_CHANGED_FAIL",
    "PERSON_OR_POSE_REGISTRATION_FAIL",
    "GARMENT_OR_HUMAN_VISUAL_FAIL",
)
EXPECTED_DISTRIBUTION = {
    "1024x1536": 5, "1166x1349": 1, "1168x1346": 1, "1168x1347": 1,
    "1173x1341": 1, "1179x1334": 1, "1348x1167": 1, "1349x1166": 34,
    "1350x1165": 3,
}
EXPECTED_COUNTS = {
    "REGISTERED_SIMILARITY_PASS_CANDIDATE": 37,
    "REGISTRATION_EVIDENCE_INSUFFICIENT": 3,
    "AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL": 7,
    "BACKGROUND_GEOMETRY_CHANGED_FAIL": 1,
    "PERSON_OR_POSE_REGISTRATION_FAIL": 0,
    "GARMENT_OR_HUMAN_VISUAL_FAIL": 0,
}
EXPECTED_NO_VALID_CELLS = ["O01/slot_04", "O01/slot_07", "O03/slot_06"]
EXPECTED_DOMINANT_GAPS = [
    "O01/slot_04", "O01/slot_07", "O03/slot_06", "O03/slot_07",
    "O04/slot_05", "O04/slot_06",
]

PROTOCOL_PATH = RISK / "subject00_attempt001_native_landscape_registration_protocol_20260726.json"
INVENTORY_PATH = RISK / "subject00_attempt001_native_landscape_full_inventory_20260726.json"
RESOLUTION_PATH = RISK / "subject00_attempt001_resolution_distribution_20260726.json"
COHORTS_PATH = RISK / "subject00_attempt001_exact_resolution_cohorts_20260726.json"
PROVENANCE_PATH = RISK / "subject00_attempt001_provenance_binding_audit_20260726.json"
REGISTRATION_PATH = RISK / "subject00_attempt001_background_registration_metrics_20260726.json"
CAMERA_PATH = RISK / "subject00_attempt001_camera_intrinsics_candidates_20260726.json"
COVERAGE_PATH = RISK / "subject00_attempt001_garment_slot_coverage_20260726.json"
REVIEW_PATH = RISK / "subject00_attempt001_native_landscape_human_review_registry_20260726.json"
SUMMARY_PATH = RISK / "subject00_attempt001_native_landscape_registration_final_summary_20260726.json"
HANDOFF_PATH = ROOT / "project_control_handoff" / "subject00_attempt001_native_landscape_registration_audit_handoff_20260726.json"
TESTS_PATH = RISK / "subject00_attempt001_native_landscape_registration_tests_20260726.json"
EXTERNAL_TESTS_PATH = AUDIT_ROOT / "07_final_summary" / "attempt001_native_landscape_registration_tests.json"
BASELINE_PATH = AUDIT_ROOT / "00_provenance" / "attempt_immutability_baseline.json"
EXTERNAL_PHOTOMETRIC_PATH = AUDIT_ROOT / "02_registration_metrics" / "attempt001_background_photometric_metrics.json"
EXTERNAL_REVIEW_PATH = AUDIT_ROOT / "06_human_review" / "attempt001_human_review_manifest.json"

DUAL_PATHS = {
    INVENTORY_PATH: AUDIT_ROOT / "01_inventory" / "attempt001_full_inventory.json",
    RESOLUTION_PATH: AUDIT_ROOT / "01_inventory" / "attempt001_resolution_distribution.json",
    COHORTS_PATH: AUDIT_ROOT / "01_inventory" / "attempt001_exact_resolution_cohorts.json",
    PROVENANCE_PATH: AUDIT_ROOT / "01_inventory" / "attempt001_provenance_binding_audit.json",
    REGISTRATION_PATH: AUDIT_ROOT / "02_registration_metrics" / "attempt001_background_registration_metrics.json",
    CAMERA_PATH: AUDIT_ROOT / "03_camera_candidates" / "attempt001_camera_intrinsics_candidates.json",
    COVERAGE_PATH: AUDIT_ROOT / "04_coverage" / "attempt001_garment_slot_coverage_matrix.json",
    REVIEW_PATH: EXTERNAL_REVIEW_PATH,
    SUMMARY_PATH: AUDIT_ROOT / "07_final_summary" / "attempt001_native_landscape_registration_final_summary.json",
}


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
    value = dict(payload)
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def run(*command: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        list(command), cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def assert_true(value: Any, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: {actual!r} != {expected!r}")


def expected_request_ids() -> list[str]:
    return [
        f"subject00_{garment}_slot{slot[-2:]}_cand{candidate:02d}"
        for garment in GARMENTS for slot in SLOT_CAMERAS for candidate in range(2)
    ]


def current_inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        item.relative_to(root).as_posix(): (item.stat().st_size, file_sha256(item))
        for item in sorted(path for path in root.rglob("*") if path.is_file())
    }


def frozen_inventory(snapshot: dict[str, Any]) -> dict[str, tuple[int, str]]:
    return {item["relative_path"]: (item["bytes"], item["sha256"]) for item in snapshot["files"]}


def changed_paths() -> list[str]:
    tracked = run("git", "diff", "--name-only", SOURCE_HEAD, "--").splitlines()
    untracked = run("git", "ls-files", "--others", "--exclude-standard").splitlines()
    return sorted(set(path for path in tracked + untracked if path))


def image_size_and_verify(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        size = image.size
        image.verify()
    return size


def verify_json_hashes(paths: list[Path]) -> None:
    for path in paths:
        payload = load_json(path)
        assert_equal(payload.get("content_sha256"), canonical_sha256(payload), f"bad content hash {path.name}")


def expected_primary(metric: dict[str, Any]) -> str:
    gate = metric["gate_results"]
    if not gate["feature_evidence"] or not gate["similarity_stable"]:
        return "REGISTRATION_EVIDENCE_INSUFFICIENT"
    if not gate["transformed_source_person_bounds"]:
        return "PERSON_OR_POSE_REGISTRATION_FAIL"
    geometry = all(gate[name] for name in (
        "similarity_stable", "scale_anisotropy", "rotation", "shear", "reprojection",
        "homography_diagnostic",
    ))
    if not geometry:
        return "AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL"
    if not gate["background_structure"]:
        return "BACKGROUND_GEOMETRY_CHANGED_FAIL"
    return "REGISTERED_SIMILARITY_PASS_CANDIDATE"


def main() -> int:
    protocol = load_json(PROTOCOL_PATH)
    inventory = load_json(INVENTORY_PATH)
    resolution = load_json(RESOLUTION_PATH)
    cohorts = load_json(COHORTS_PATH)
    provenance = load_json(PROVENANCE_PATH)
    registration = load_json(REGISTRATION_PATH)
    camera = load_json(CAMERA_PATH)
    coverage = load_json(COVERAGE_PATH)
    review = load_json(REVIEW_PATH)
    summary = load_json(SUMMARY_PATH)
    handoff = load_json(HANDOFF_PATH)
    baseline = load_json(BASELINE_PATH)
    photometric = load_json(EXTERNAL_PHOTOMETRIC_PATH)
    cloud_manifest = load_json(CLOUD_MANIFEST)
    calibration = load_json(CALIBRATION)
    paths = changed_paths()
    expected_ids = expected_request_ids()
    records = {item["request_id"]: item for item in inventory["records"]}
    metrics = {item["request_id"]: item for item in registration["records"]}
    photos = {item["request_id"]: item for item in photometric["records"]}
    review_records = {item["request_id"]: item for item in review["records"]}
    cloud_files = {item["relative_path"]: item for item in cloud_manifest["files"]}
    checks: list[dict[str, Any]] = []

    def check(name: str, function: Callable[[], None]) -> None:
        try:
            function()
            checks.append({"name": name, "status": "PASS"})
        except Exception as error:  # noqa: BLE001 - retain every failed contract gate
            checks.append({"name": name, "status": "FAIL", "detail": f"{type(error).__name__}: {error}"})

    check("source_branch", lambda: assert_equal(run("git", "branch", "--show-current", cwd=SOURCE_WORKTREE), SOURCE_BRANCH, "source branch"))
    check("source_head", lambda: assert_equal(run("git", "rev-parse", "HEAD", cwd=SOURCE_WORKTREE), SOURCE_HEAD, "source HEAD"))
    check("source_worktree_clean", lambda: assert_equal(run("git", "status", "--short", cwd=SOURCE_WORKTREE), "", "source worktree dirty"))
    check("target_branch", lambda: assert_equal(run("git", "branch", "--show-current"), BRANCH, "target branch"))
    check("source_head_ancestor", lambda: assert_true(
        subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT).returncode == 0,
        "source HEAD is not an ancestor",
    ))
    check("attempt_001_immutable", lambda: assert_equal(current_inventory(ATTEMPTS["attempt_001"]), frozen_inventory(baseline["attempt_001"]), "attempt_001 changed"))
    check("attempt_002_immutable", lambda: assert_equal(current_inventory(ATTEMPTS["attempt_002"]), frozen_inventory(baseline["attempt_002"]), "attempt_002 changed"))
    check("attempt_003_immutable", lambda: assert_equal(current_inventory(ATTEMPTS["attempt_003"]), frozen_inventory(baseline["attempt_003"]), "attempt_003 changed"))

    check("task_id_consistent", lambda: assert_true(all(
        item["task_id"] == TASK_ID for item in (
            protocol, inventory, resolution, cohorts, provenance, registration, camera, coverage,
            review, summary, handoff, baseline, photometric,
        )
    ), "task ID mismatch"))
    check("request_count_48", lambda: assert_equal(inventory["request_count"], 48, "request count"))
    check("output_count_48", lambda: assert_equal(inventory["output_count"], 48, "output count"))
    check("expected_request_set", lambda: assert_equal(sorted(records), sorted(expected_ids), "request ID set"))
    check("metric_request_set", lambda: assert_equal(sorted(metrics), sorted(expected_ids), "metric request ID set"))
    check("review_request_set", lambda: assert_equal(sorted(review_records), sorted(expected_ids), "review request ID set"))

    parsed_sizes: dict[str, tuple[int, int]] = {}

    def verify_outputs() -> None:
        for request_id, record in records.items():
            output = Path(record["generated_output_path"])
            assert_true(output.is_file(), f"missing output {request_id}")
            assert_equal(output.suffix.lower(), ".png", f"non-PNG output {request_id}")
            parsed_sizes[request_id] = image_size_and_verify(output)
            assert_equal(parsed_sizes[request_id], (record["output_width"], record["output_height"]), f"PNG size {request_id}")

    check("all_png_files_readable", verify_outputs)
    check("png_parse_pass_count_48", lambda: assert_true(
        inventory["png_parse_pass_count"] == summary["png_parse_pass_count"] == 48
        and all(item["png_parse_status"] == "PASS" for item in records.values()),
        "PNG parse state mismatch",
    ))
    check("output_sha256_all_match", lambda: assert_true(all(
        file_sha256(Path(item["generated_output_path"])) == item["generated_output_sha256"]
        for item in records.values()
    ), "output SHA mismatch"))
    check("no_duplicate_output_sha", lambda: assert_true(
        not inventory["duplicate_sha_groups"]
        and len({item["generated_output_sha256"] for item in records.values()}) == 48,
        "duplicate output SHA detected",
    ))
    check("provider_contract", lambda: assert_true(all(
        item["provider"] == "CODEX_MANAGED_IMAGE_EDIT" for item in records.values()
    ), "provider mismatch"))
    check("retry_count_contract", lambda: assert_equal(
        Counter(item["retry_count"] for item in records.values()), Counter({0: 47, 1: 1}), "retry distribution",
    ))
    check("postprocessing_not_fabricated", lambda: assert_true(
        all(item["postprocessing_count"] == "NOT_RECORDED" for item in records.values())
        and not any(item["hidden_postprocessing_replacement_detected"] for item in records.values()),
        "postprocessing evidence was changed",
    ))

    check("provenance_binding_48", lambda: assert_true(
        provenance["request_count"] == provenance["pass_count"] == summary["provenance_binding_pass_count"] == 48
        and all(item["binding_status"] == "PASS" and all(item["binding_checks"].values()) for item in provenance["records"]),
        "provenance binding mismatch",
    ))
    check("source_rgb_sha256", lambda: assert_true(all(
        file_sha256(Path(item["original_condition_path"])) == item["original_condition_sha256"]
        for item in records.values()
    ), "source RGB SHA mismatch"))
    check("source_rgb_shared_by_slot", lambda: assert_true(all(
        len({records[f"subject00_{garment}_slot{slot[-2:]}_cand{candidate:02d}"]["original_condition_sha256"]
             for garment in GARMENTS for candidate in range(2)}) == 1
        for slot in SLOT_CAMERAS
    ), "source RGB differs within a slot"))

    def verify_mask_shas() -> None:
        for item in records.values():
            member = item["original_condition_mask_member"]
            relative = member.removeprefix("subject00/")
            assert_equal(item["original_condition_mask_archive"], str(ARCHIVE), "mask archive")
            assert_equal(item["original_condition_mask_sha256"], cloud_files[relative]["sha256"], f"mask SHA {item['request_id']}")

    check("official_mask_sha256", verify_mask_shas)
    check("calibration_sha256", lambda: assert_true(
        file_sha256(CALIBRATION) == provenance["source_calibration_sha256"] == CALIBRATION_SHA256,
        "calibration SHA mismatch",
    ))
    check("slot_camera_mapping", lambda: assert_true(all(
        item["camera"] == SLOT_CAMERAS[item["slot"]] and item["pose_frame_id"] == 0
        for item in records.values()
    ), "slot/camera/frame mapping mismatch"))
    check("garment_slot_candidate_cardinality", lambda: assert_true(all(
        sum(item["garment"] == garment and item["slot"] == slot for item in records.values()) == 2
        for garment in GARMENTS for slot in SLOT_CAMERAS
    ), "garment/slot cardinality mismatch"))

    recomputed_distribution = Counter(f"{width}x{height}" for width, height in parsed_sizes.values())
    check("resolution_distribution_exact", lambda: assert_equal(dict(sorted(recomputed_distribution.items())), EXPECTED_DISTRIBUTION, "resolution distribution"))
    check("resolution_registry_consistent", lambda: assert_true(
        resolution["exact_distribution"] == EXPECTED_DISTRIBUTION
        and resolution["historical_distribution"] == EXPECTED_DISTRIBUTION
        and resolution["historical_match"]
        and resolution["parse_failure_count"] == 0,
        "resolution registry mismatch",
    ))
    check("landscape_portrait_counts", lambda: assert_true(
        resolution["landscape_count"] == 38
        and resolution["exact_1024x1536_count"] == 5
        and resolution["portrait_incompatible_non_native_count"] == 5,
        "orientation counts mismatch",
    ))

    raw_cells: dict[str, set[str]] = defaultdict(set)
    pass_cells: dict[str, set[str]] = defaultdict(set)
    for request_id, item in records.items():
        label = f"{item['output_width']}x{item['output_height']}"
        cell = f"{item['garment']}/{item['slot']}"
        raw_cells[label].add(cell)
        if metrics[request_id]["primary_classification"] == "REGISTERED_SIMILARITY_PASS_CANDIDATE":
            pass_cells[label].add(cell)

    check("cohort_count_9", lambda: assert_equal(cohorts["cohort_count"], 9, "cohort count"))
    check("cohort_counts_recomputed", lambda: assert_true(all(
        item["output_count"] == EXPECTED_DISTRIBUTION[item["resolution_label"]]
        and item["garment_slot_cell_count"] == len(raw_cells[item["resolution_label"]])
        and item["machine_pass_cell_count"] == len(pass_cells[item["resolution_label"]])
        for item in cohorts["cohorts"]
    ), "cohort count or cell coverage mismatch"))
    check("dominant_cohort", lambda: assert_true(
        cohorts["dominant_resolution"] == summary["dominant_resolution"] == "1349x1166"
        and cohorts["dominant_output_count"] == summary["dominant_cohort_output_count"] == 34
        and cohorts["dominant_garment_slot_coverage"] == summary["dominant_cohort_garment_slot_coverage"] == 20
        and cohorts["dominant_machine_pass_cell_coverage"] == 18,
        "dominant cohort mismatch",
    ))
    check("no_complete_single_resolution_cohort", lambda: assert_true(
        not cohorts["raw_complete_single_resolution_cohorts"]
        and not cohorts["reliable_complete_single_resolution_cohorts"]
        and not cohorts["complete_single_resolution_cohort_found"]
        and not summary["complete_single_resolution_cohort_found"],
        "complete cohort unexpectedly found",
    ))

    check("registration_record_count_48", lambda: assert_equal(registration["audit_count"], 48, "registration count"))
    check("all_registration_models_computed", lambda: assert_true(all(
        item[model]["status"] == "PASS_COMPUTED"
        for item in metrics.values()
        for model in ("uniform_scale_translation", "similarity", "axis_scale_translation", "affine", "homography")
    ), "registration model missing"))
    check("feature_counts_recorded", lambda: assert_true(all(
        item["background_feature_count"] >= 0 and item["output_feature_count"] >= 0 and item["match_count"] >= 0
        for item in metrics.values()
    ), "feature counts missing"))
    check("person_exclusion_evidence", lambda: assert_true(all(
        len(item["person_pose_metrics"]["person_bbox_source"]) == 4
        and item["photometric"]["valid_background_pixel_count"] > 0
        and 0 < item["photometric"]["valid_background_fraction"] < 1
        for item in metrics.values()
    ), "person exclusion evidence incomplete"))
    check("mask_threshold_and_dilation_23", lambda: assert_true(
        protocol["background_mask"]["foreground_threshold"] == 128
        and protocol["background_mask"]["person_dilation_radius_px_for_1330x1150"] == 23
        and max(20, round(0.02 * min(1330, 1150))) == 23
        and protocol["background_mask"]["image_border_exclusion_px"] == 8,
        "mask/dilation protocol mismatch",
    ))
    check("protocol_frozen_thresholds", lambda: assert_true(
        protocol["threshold_adjustment_policy"] == "NO_PER_IMAGE_OR_POST_RESULT_THRESHOLD_ADJUSTMENT"
        and protocol["models"]["ransac_reprojection_threshold_px"] == 3.0
        and protocol["models"]["ransac_max_iterations"] == 5000
        and protocol["models"]["ransac_confidence"] == 0.999,
        "registration protocol drift",
    ))

    gate = protocol["machine_similarity_gate"]
    structure_gate = protocol["background_structure_gate"]
    machine_pass = [item for item in metrics.values() if item["primary_classification"] == "REGISTERED_SIMILARITY_PASS_CANDIDATE"]
    check("machine_pass_gate_vector", lambda: assert_true(all(
        all(item["gate_results"].values()) for item in machine_pass
    ), "machine pass has a failed gate"))
    check("machine_pass_inlier_gate", lambda: assert_true(all(
        item["similarity"]["inlier_count"] >= protocol["models"]["minimum_similarity_inliers"]
        and item["similarity"]["inlier_ratio"] >= gate["minimum_inlier_ratio"]
        for item in machine_pass
    ), "machine-pass inlier gate failed"))
    check("machine_pass_reprojection_gate", lambda: assert_true(all(
        item["median_reprojection_error_px"] <= gate["maximum_median_reprojection_error_px"]
        and item["p95_reprojection_error_px"] <= gate["maximum_p95_reprojection_error_px"]
        for item in machine_pass
    ), "machine-pass reprojection gate failed"))
    check("machine_pass_scale_rotation_shear_gates", lambda: assert_true(all(
        abs(item["estimated_scale_x"] / item["estimated_scale_y"] - 1) <= gate["maximum_abs_scale_ratio_minus_one"]
        and abs(item["rotation_degrees"]) <= gate["maximum_abs_rotation_degrees"]
        and abs(item["normalized_shear"]) <= gate["maximum_abs_normalized_shear"]
        for item in machine_pass
    ), "machine-pass scale/rotation/shear gate failed"))
    check("machine_pass_homography_person_gates", lambda: assert_true(all(
        item["homography_relative_similarity_error_improvement"] <= gate["maximum_homography_relative_median_error_improvement"]
        and item["transformed_source_person_bounds"]["pass"]
        for item in machine_pass
    ), "machine-pass homography/person gate failed"))
    check("photometric_record_count_48", lambda: assert_true(
        photometric["audit_count"] == 48 and sorted(photos) == sorted(expected_ids),
        "photometric record count mismatch",
    ))
    check("photometric_dual_record_consistency", lambda: assert_true(all(
        all(math.isclose(float(photos[request_id][key]), float(metrics[request_id]["photometric"][key]), rel_tol=0, abs_tol=1e-12)
            for key in ("masked_ssim", "edge_alignment_f1", "structural_difference_area_fraction"))
        for request_id in expected_ids
    ), "photometric copies disagree"))
    check("machine_pass_background_structure_gate", lambda: assert_true(all(
        item["photometric"]["masked_ssim"] >= structure_gate["minimum_masked_ssim"]
        and item["photometric"]["edge_alignment_f1"] >= structure_gate["minimum_edge_alignment_f1"]
        and item["photometric"]["structural_difference_area_fraction"] <= structure_gate["maximum_structural_difference_area_fraction"]
        for item in machine_pass
    ), "machine-pass background structure gate failed"))
    check("primary_classification_recomputed", lambda: assert_true(all(
        item["primary_classification"] == expected_primary(item) for item in metrics.values()
    ), "primary classification mismatch"))
    check("classification_counts", lambda: assert_true(
        Counter(item["primary_classification"] for item in metrics.values()) == Counter(EXPECTED_COUNTS)
        and registration["classification_counts"] == EXPECTED_COUNTS
        and summary["classification_counts"] == EXPECTED_COUNTS
        and sum(summary["classification_counts"].values()) == 48,
        "classification counts mismatch",
    ))
    check("machine_pass_not_visual_pass", lambda: assert_true(
        not protocol["human_visual_policy"]["machine_pass_is_visual_pass"]
        and all(item["human_visual_decision"] is None for item in metrics.values()),
        "machine result promoted to visual decision",
    ))

    def verify_camera_candidates() -> None:
        assert_equal(camera["candidate_count"], 37, "camera candidate count")
        assert_equal(camera["invalid_count"], 11, "camera invalid count")
        assert_equal(camera["materialized_count"], 0, "materialized camera count")
        assert_equal({item["request_id"] for item in camera["records"]}, {item["request_id"] for item in machine_pass}, "camera candidate set")
        for item in camera["records"]:
            metric = metrics[item["request_id"]]
            source_k_values = calibration[item["camera"]]["K"]
            source_k = (
                [source_k_values[0:3], source_k_values[3:6], source_k_values[6:9]]
                if source_k_values and not isinstance(source_k_values[0], list)
                else source_k_values
            )
            transform = item["transform"]
            expected = {
                "fx": transform["scale_x"] * source_k[0][0],
                "fy": transform["scale_y"] * source_k[1][1],
                "cx": transform["scale_x"] * source_k[0][2] + transform["translation_x"],
                "cy": transform["scale_y"] * source_k[1][2] + transform["translation_y"],
            }
            for key, value in expected.items():
                assert_true(math.isclose(item["derived_intrinsics"][key], value, rel_tol=0, abs_tol=1e-9), f"camera {key} {item['request_id']}")
            assert_equal(item["width_out"], records[item["request_id"]]["output_width"], "camera width")
            assert_equal(item["height_out"], records[item["request_id"]]["output_height"], "camera height")
            assert_true(item["camera_compatible"] and not item["materialized"], "camera state")
            assert_equal(item["source_calibration_sha256"], CALIBRATION_SHA256, "camera calibration SHA")
            assert_equal(transform["scale_x"], metric["axis_scale_translation"]["scale_x"], "camera scale_x")

    check("camera_intrinsics_derivation", verify_camera_candidates)

    recomputed_cell_counts: dict[str, int] = {}
    for garment in GARMENTS:
        for slot in SLOT_CAMERAS:
            cell = f"{garment}/{slot}"
            request_ids = [f"subject00_{garment}_slot{slot[-2:]}_cand{candidate:02d}" for candidate in range(2)]
            recomputed_cell_counts[cell] = sum(metrics[item]["primary_classification"] == "REGISTERED_SIMILARITY_PASS_CANDIDATE" for item in request_ids)

    check("coverage_24_cells", lambda: assert_true(
        coverage["cell_count"] == len(coverage["records"]) == 24
        and {item["cell"] for item in coverage["records"]} == set(recomputed_cell_counts),
        "coverage cell set mismatch",
    ))
    check("coverage_candidate_counts", lambda: assert_true(all(
        item["machine_pass_candidate_count"] == recomputed_cell_counts[item["cell"]]
        for item in coverage["records"]
    ), "coverage candidate count mismatch"))
    check("machine_pass_21_cell_coverage", lambda: assert_true(
        sum(count > 0 for count in recomputed_cell_counts.values())
        == coverage["machine_pass_24_cell_coverage"]
        == summary["machine_pass_24_cell_coverage"]
        == 21,
        "machine-pass cell coverage mismatch",
    ))
    check("no_valid_candidate_cells", lambda: assert_true(
        [cell for cell, count in recomputed_cell_counts.items() if count == 0]
        == coverage["no_valid_candidate_cells"]
        == summary["no_valid_candidate_cells"]
        == EXPECTED_NO_VALID_CELLS,
        "no-valid-candidate cells mismatch",
    ))
    check("minimum_targeted_rerun_requests", lambda: assert_true(
        coverage["dominant_resolution_machine_pass_missing_cells"] == EXPECTED_DOMINANT_GAPS
        and coverage["minimum_targeted_rerun_request_count"] == len(EXPECTED_DOMINANT_GAPS) == 6
        and [item["cell"] for item in coverage["minimum_targeted_rerun_requests"]] == EXPECTED_DOMINANT_GAPS
        and all(item["target_resolution"] == "1349x1166" and item["minimum_new_candidate_count"] == 1 and not item["generation_authorized"]
                for item in coverage["minimum_targeted_rerun_requests"]),
        "targeted rerun contract mismatch",
    ))

    null_fields = {
        "background_geometry", "background_perspective", "camera_direction", "feet_complete",
        "final_decision", "garment_boundary", "garment_correctness", "hands_complete",
        "identity_consistency", "pose_preservation", "review_timestamp", "reviewer", "silhouette",
        "single_person", "subject_center", "subject_scale", "visible_generation_artifacts",
        "accepted", "teacher_target",
    }
    check("human_review_record_count_48", lambda: assert_true(
        review["record_count"] == len(review["records"]) == 48 and review["status"] == "PENDING_USER_REVIEW",
        "review registry count/status mismatch",
    ))
    check("human_review_fields_null", lambda: assert_true(all(
        all(item[field] is None for field in null_fields) for item in review["records"]
    ), "human review decision was prefilled"))
    check("accepted_teacher_zero", lambda: assert_true(
        review["accepted_count"] == review["teacher_target_count"]
        == summary["accepted_count"] == summary["teacher_target_count"] == 0,
        "accepted or Teacher target count is nonzero",
    ))

    request_panels = sorted((AUDIT_ROOT / "05_contact_sheets" / "requests").glob("*.png"))
    overview_panels = sorted((AUDIT_ROOT / "05_contact_sheets").glob("*.png"))
    check("request_contact_sheet_count_48", lambda: assert_equal(len(request_panels), 48, "request panel count"))
    check("overview_contact_sheet_count_7", lambda: assert_equal(len(overview_panels), 7, "overview panel count"))
    check("contact_sheet_paths_bound", lambda: assert_true(
        set(Path(path) for path in summary["contact_sheet_paths"].values()) == set(overview_panels)
        and all(Path(item["request_contact_sheet_path"]).is_file() for item in review["records"]),
        "contact sheet path binding mismatch",
    ))
    check("all_contact_sheets_readable", lambda: assert_true(all(
        image_size_and_verify(path)[0] > 0 and image_size_and_verify(path)[1] > 0
        for path in request_panels + overview_panels
    ), "unreadable contact sheet"))

    check("dual_json_copies_exact", lambda: assert_true(all(
        file_sha256(repo_path) == file_sha256(external_path) for repo_path, external_path in DUAL_PATHS.items()
    ), "Git/external JSON copy mismatch"))
    hash_paths = [path for path in DUAL_PATHS] + [HANDOFF_PATH, BASELINE_PATH, EXTERNAL_PHOTOMETRIC_PATH]
    check("json_content_hashes", lambda: verify_json_hashes(hash_paths))
    check("protocol_sha_binding", lambda: assert_true(
        file_sha256(PROTOCOL_PATH) == summary["protocol_sha256"] == registration["protocol_sha256"] == provenance["protocol_sha256"],
        "protocol SHA binding mismatch",
    ))

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    check("no_git_images", lambda: assert_true(
        not any(Path(path).suffix.lower() in image_extensions for path in paths), "image entered Git changes"
    ))
    check("no_generation_or_external_calls", lambda: assert_true(
        summary["new_generation_calls"] == summary["external_api_calls"] == 0,
        "generation or external API call recorded",
    ))
    check("no_api_keys_or_cloud_images", lambda: assert_true(
        summary["api_key_reads"] == summary["cloud_image_writes"] == 0,
        "API key or cloud image write recorded",
    ))
    check("no_training_code_modification", lambda: assert_true(
        not any(path.startswith(("scene/", "utils/", "config/")) or Path(path).name.startswith("train") for path in paths),
        "training code changed",
    ))
    check("no_paper_modification", lambda: assert_true(
        summary["paper_modifications"] == 0
        and not any(path.startswith("docs/PAPER/") or Path(path).suffix.lower() in {".tex", ".bib"} for path in paths),
        "paper source changed",
    ))
    check("attempt_mutation_counts_zero", lambda: assert_true(
        summary["attempt_001_mutations"] == summary["attempt_002_mutations"] == summary["attempt_003_mutations"] == 0,
        "attempt mutation count is nonzero",
    ))
    check("formal_base_and_bulk_state", lambda: assert_true(
        summary["formal_base_status"] == "PENDING"
        and summary["remaining_bulk_generation_authorization"] == "DENIED"
        and summary["subject00_paper_positive_claim_count"] == 0,
        "formal base/bulk/paper state mismatch",
    ))
    check("feasibility_conclusions", lambda: assert_true(
        summary["registration_feasibility"] == "PARTIALLY_FEASIBLE"
        and summary["exact_resolution_cohort_feasibility"] == "PARTIAL_SINGLE_RESOLUTION_COHORT"
        and summary["per_view_resolution_requirement"] == "UNRESOLVED"
        and summary["dataset_salvage_classification"] == "PARTIAL_SALVAGE_TARGETED_RERUN_CANDIDATE",
        "feasibility conclusion mismatch",
    ))
    check("final_classification_unique", lambda: assert_true(
        summary["final_classification"] == handoff["final_classification"] == FINAL_CLASSIFICATION,
        "final classification mismatch",
    ))
    check("next_task_unique", lambda: assert_true(
        summary["next_task"] == handoff["next_task"] == NEXT_TASK,
        "next task mismatch",
    ))
    check("paper_final_false", lambda: assert_true(all(
        item["paper_final"] is False for item in (
            protocol, inventory, resolution, cohorts, provenance, registration, camera, coverage,
            review, summary, handoff, baseline, photometric,
        )
    ), "PAPER_FINAL changed"))

    failed = [item for item in checks if item["status"] != "PASS"]
    payload = {
        "schema_version": "canondressgs.subject00.attempt001_native_landscape_registration_tests.v1",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "branch": BRANCH,
        "test_count": len(checks),
        "pass_count": len(checks) - len(failed),
        "fail_count": len(failed),
        "result": "PASS" if not failed else "FAIL",
        "checks": checks,
        "attempt_tree_sha256": {
            name: baseline[name]["tree_sha256"] for name in ("attempt_001", "attempt_002", "attempt_003")
        },
        "final_classification": FINAL_CLASSIFICATION if not failed else "SUBJECT00_NATIVE_LANDSCAPE_AUDIT_CONTRACT_VIOLATION",
        "paper_final": False,
        "next_task": NEXT_TASK if not failed else "REPAIR_SUBJECT00_NATIVE_LANDSCAPE_AUDIT_CONTRACT",
    }
    write_json(TESTS_PATH, payload)
    write_json(EXTERNAL_TESTS_PATH, payload)
    print(json.dumps({
        "result": payload["result"],
        "test_count": payload["test_count"],
        "pass_count": payload["pass_count"],
        "fail_count": payload["fail_count"],
        "tests_path": str(TESTS_PATH),
        "external_tests_path": str(EXTERNAL_TESTS_PATH),
        "failed": failed,
    }, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
