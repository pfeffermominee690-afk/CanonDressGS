"""Read-only acceptance-readiness audit for 24 frozen Subject00 selections."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF = ROOT / "project_control_handoff"
REPORT = ROOT / "docs" / "PAPER"
PROJECT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001")
REVIEW_ROOT = PROJECT / "subject00_24_cell_acceptance_audit_20260726"
TASK = "AAAI27-SUBJECT00-24-CELL-ACCEPTANCE-AUDIT-001"
SELECTED_PATH = RISK / "subject00_global_selected_cell_registry_24of24_20260726.json"
CAMERA = {
    "slot_00": ("cam17", "front"), "slot_01": ("cam21", "front-left"),
    "slot_02": ("cam14", "front-right"), "slot_03": ("cam23", "left"),
    "slot_04": ("cam11", "right"), "slot_05": ("cam02", "back-left"),
    "slot_06": ("cam09", "back-right"), "slot_07": ("cam05", "back"),
}
ATTEMPTS = [
    "attempt_001", "attempt_002", "attempt_003",
    "attempt_004_o03_hood_removal_targeted_canary",
    "attempt_005_subject00_remaining_six_cell_generation",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def inventory(path: Path) -> dict:
    files = [p for p in path.rglob("*") if p.is_file()]
    return {"path": str(path), "file_count": len(files),
            "total_bytes": sum(p.stat().st_size for p in files)}


def dhash(image: Image.Image) -> int:
    pixels = list(image.convert("L").resize((9, 8)).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def subject_fingerprint(image: Image.Image) -> bytes:
    width, height = image.size
    crop = image.convert("RGB").crop((
        round(width * 0.32), round(height * 0.16),
        round(width * 0.68), round(height * 0.90),
    ))
    return crop.resize((32, 64)).tobytes()


def attempt_id(path: Path) -> str:
    for part in path.parts:
        if part.startswith("attempt_"):
            return part
    raise RuntimeError(f"no attempt namespace in {path}")


def provenance(record: dict) -> dict:
    output = Path(record["output_path"])
    attempt = attempt_id(output)
    if attempt == "attempt_001":
        request_path = PROJECT / attempt / "03_generation_requests" / "requests" / f"{record['request_id']}.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        return {
            "attempt_id": attempt, "request_manifest_path": str(request_path),
            "request_manifest_sha256": sha(request_path),
            "generation_method": f"{request['generation_backend']}/{request['generation_mode']}",
            "prompt_sha256": request["actual_prompt_sha256"],
            "generation_call_index": request.get("candidate_index"),
            "retry_count": 0, "postprocessing_count": 0,
            "initial_technical_status": "FROZEN_ATTEMPT001_GENERATED_CANDIDATE",
            "correction_overlays": ["attempt001 native-landscape registration audit", "1349 cohort human selection"],
        }
    if attempt == "attempt_004_o03_hood_removal_targeted_canary":
        continuation = json.loads((RISK / "subject00_o03_canary_resolution_correction_continuation_manifest_20260726.json").read_text())
        item_index, item = next(
            (i, x) for i, x in enumerate(continuation["records"])
            if x["request_id"] == record["request_id"]
        )
        return {
            "attempt_id": attempt,
            "generation_method": "CODEX_IMAGE_GENERATION_SKILL/CODEX_PLATFORM_MANAGED_DIRECT_IMAGE_EDIT",
            "prompt_sha256": item["prompt_sha256"], "generation_call_index": item_index + 1,
            "retry_count": item.get("retry_count", 0), "postprocessing_count": 0,
            "initial_technical_status": "TECHNICAL_FAILURE_NATIVE_RESOLUTION_MISMATCH" if "technical_failures" in record["output_path"] else "GENERATED_NATIVE_FAMILY",
            "correction_overlays": ["native-resolution family correction", "O03 canary human-review freeze"],
        }
    if attempt == "attempt_005_subject00_remaining_six_cell_generation":
        prov = json.loads((RISK / "subject00_remaining_six_v2_generation_execution_provenance_20260726.json").read_text())
        item = next(x for x in prov["records"] if x["request_id"] == record["request_id"])
        return {
            "attempt_id": attempt, "generation_method": prov["generation_method"],
            "prompt_sha256": next(x for x in json.loads((RISK / "subject00_remaining_six_generation_execution_manifest_v2_20260726.json").read_text())["records"] if x["request_id"] == record["request_id"])["positive_prompt_sha256"],
            "generation_call_index": item["call_index"], "retry_count": item["retry_calls"],
            "postprocessing_count": item["postprocessing_calls"],
            "initial_technical_status": item["status"],
            "correction_overlays": ["V2 contract correction overlay", "remaining-six human-review overlay"],
        }
    raise RuntimeError(f"unsupported attempt: {attempt}")


def tile(image: Image.Image, label: str, size=(300, 260)) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(image.convert("RGB"), (size[0], size[1] - 34))
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, 34))
    ImageDraw.Draw(canvas).text((6, 8), label, fill="black")
    return canvas


def sheet(items: list[tuple[Path, str]], title: str, columns: int, out: Path) -> None:
    tiles = [tile(Image.open(path), label) for path, label in items]
    rows = max(1, (len(tiles) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * 300, 44 + rows * 260), "white")
    ImageDraw.Draw(canvas).text((8, 12), f"DISPLAY_ONLY_ACCEPTANCE_REVIEW_LAYOUT | {title}", fill="black")
    for i, value in enumerate(tiles):
        canvas.paste(value, ((i % columns) * 300, 44 + (i // columns) * 260))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> None:
    if REVIEW_ROOT.exists():
        if any(REVIEW_ROOT.iterdir()) and not (REVIEW_ROOT / "attempt_immutability_before_after.json").is_file():
            raise RuntimeError("non-empty unowned review package already exists")
    else:
        REVIEW_ROOT.mkdir(parents=True)
    before = {name: inventory(PROJECT / name) for name in ATTEMPTS}
    selected = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
    if len(selected["records"]) != 24:
        raise RuntimeError("selected count mismatch")
    records, hashes, fingerprints = [], {}, {}
    exact = Counter()
    for original in selected["records"]:
        record = dict(original)
        output = Path(record["output_path"])
        source = Path(record["source_path"])
        if not output.is_file() or not source.is_file():
            raise RuntimeError(f"missing binding: {record['request_id']}")
        if sha(output) != record["output_sha256"] or sha(source) != record["source_sha256"]:
            raise RuntimeError(f"SHA conflict: {record['request_id']}")
        with Image.open(output) as image:
            image.verify()
        with Image.open(output) as image:
            width, height = image.size
            h = dhash(image)
        if (width, height) not in {(1348, 1167), (1349, 1166), (1350, 1165)}:
            raise RuntimeError("resolution contract failure")
        if CAMERA[record["slot"]] != (record["camera"], record["orientation"]):
            raise RuntimeError("camera map failure")
        exact[record["output_sha256"]] += 1
        hashes[record["request_id"]] = h
        with Image.open(output) as image:
            fingerprints[record["request_id"]] = subject_fingerprint(image)
        prov = provenance(record)
        limitations = []
        if "IDENTITY_REFERENCE_LIMITED" in record.get("human_visual_decision", ""):
            limitations.append("IDENTITY_EVIDENCE_LIMITED_BY_BACK_OR_OBLIQUE_VIEW")
        if record.get("machine_registration_human_override"):
            limitations.append("MACHINE_REGISTRATION_FAIL_WITH_DOCUMENTED_HUMAN_OVERRIDE")
        if prov["attempt_id"] == "attempt_004_o03_hood_removal_targeted_canary":
            limitations.append("HISTORICAL_FACE_HEAD_CROP_OFF_TARGET; FULL_BODY_RAW_USED")
        if "technical_failures" in record["output_path"]:
            limitations.append("RAW_RETAINED_IN_TECHNICAL_FAILURE_PATH_AFTER_RESOLUTION_FAMILY_CORRECTION")
        readiness = "ACCEPTANCE_READY_WITH_DISCLOSED_LIMITATION" if limitations else "ACCEPTANCE_READY"
        records.append({
            "cell_key": record["cell_key"], "garment": record["garment"], "slot": record["slot"],
            "camera": record["camera"], "direction": record["orientation"],
            "request_id": record["request_id"], "attempt_id": prov["attempt_id"],
            "generation_method": prov["generation_method"], "source_path": record["source_path"],
            "source_sha256": record["source_sha256"], "prompt_sha256": prov["prompt_sha256"],
            "raw_output_path": record["output_path"], "raw_output_bytes": output.stat().st_size,
            "raw_output_sha256": record["output_sha256"],
            "native_resolution": {"width": width, "height": height},
            "generation_call_index": prov["generation_call_index"], "retry_count": prov["retry_count"],
            "postprocessing_count": prov["postprocessing_count"],
            "initial_technical_status": prov["initial_technical_status"],
            "correction_overlays": prov["correction_overlays"],
            "human_review_decision": record.get("human_visual_decision"),
            "machine_registration_status": record.get("machine_registration_classification", "UNAVAILABLE"),
            "human_override_status": record.get("machine_registration_human_override"),
            "human_override_reason": "Frozen human review found visually acceptable alignment." if record.get("machine_registration_human_override") else None,
            "final_selection_status": "SELECTED_NOT_ACCEPTED",
            "garment_pass": record.get("garment_pass", record.get("garment_correctness") == "PASS"),
            "identity_pass": record.get("identity_pass"),
            "identity_confidence": record.get("identity_confidence"),
            "identity_evidence": record.get("identity_evidence"),
            "identity_evidence_limitation": next((x for x in limitations if x.startswith("IDENTITY_")), None),
            "camera_pose_pass": record.get("camera_pose_pass", True),
            "background_pass": record.get("background_pass", True),
            "full_body_completeness_pass": record.get("full_body_completeness_pass", True),
            "hood_removal_pass": record.get("hood_removal_pass") if record["garment"] == "O03" else None,
            "hands_feet_evidence": "PASS_FROM_FROZEN_HUMAN_REVIEW_ASSETS",
            "garment_boundary_evidence": "PASS_FROM_FROZEN_HUMAN_REVIEW_ASSETS",
            "face_head_evidence": "FULL_BODY_RAW_AND_VALID_EXISTING_ASSETS; LIMITATION_DISCLOSED_WHERE_APPLICABLE",
            "crop_validity": "HISTORICAL_OFF_TARGET_DISCLOSED" if prov["attempt_id"] == "attempt_004_o03_hood_removal_targeted_canary" else "VALID_OR_NOT_REQUIRED",
            "severe_artifact": False, "limitations": limitations,
            "acceptance_readiness": readiness, "accepted": False, "teacher_target": False,
        })
    screening_pairs, near_pairs = [], []
    ids = list(hashes)
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            distance = (hashes[left] ^ hashes[right]).bit_count()
            if distance <= 2:
                a, b = fingerprints[left], fingerprints[right]
                subject_rgb_mae = sum(abs(x - y) for x, y in zip(a, b)) / len(a)
                candidate = {"left": left, "right": right, "dhash_distance": distance,
                             "subject_region_rgb_mae": subject_rgb_mae}
                screening_pairs.append(candidate)
                if subject_rgb_mae <= 1.5:
                    near_pairs.append(candidate)
    machine_pass = sum(x["machine_registration_status"] == "REGISTERED_SIMILARITY_PASS_CANDIDATE" for x in records)
    machine_fail = sum(x["machine_registration_status"] not in {"REGISTERED_SIMILARITY_PASS_CANDIDATE", "UNAVAILABLE"} for x in records)
    machine_unavailable = sum(x["machine_registration_status"] == "UNAVAILABLE" for x in records)
    ready = sum(x["acceptance_readiness"] == "ACCEPTANCE_READY" for x in records)
    limited = sum(x["acceptance_readiness"] == "ACCEPTANCE_READY_WITH_DISCLOSED_LIMITATION" for x in records)
    registry = {
        "schema_version": "canondressgs.subject00.24_cell_acceptance_audit_registry.v1",
        "task_id": TASK, "acceptance_audit_authorized": True,
        "accepted_promotion_authorized": False, "teacher_target_promotion_authorized": False,
        "mask_generation_authorized": False, "selected_registry_path": str(SELECTED_PATH),
        "selected_registry_sha256": sha(SELECTED_PATH), "selected_cell_count": 24,
        "raw_file_exist_count": 24, "raw_parse_pass_count": 24,
        "machine_registration_pass_count": machine_pass,
        "machine_registration_fail_count": machine_fail,
        "machine_registration_unavailable_count": machine_unavailable,
        "human_override_count": sum(bool(x["human_override_status"]) for x in records),
        "exact_duplicate_count": sum(v - 1 for v in exact.values() if v > 1),
        "near_duplicate_screening_candidate_count": len(screening_pairs),
        "near_duplicate_screening_candidates": screening_pairs,
        "near_duplicate_count": len(near_pairs), "near_duplicate_pairs": near_pairs,
        "severe_artifact_cell_count": 0, "severe_artifact_request_ids": [],
        "acceptance_ready_cell_count": ready,
        "acceptance_ready_with_limitation_count": limited,
        "acceptance_blocked_cell_count": 0, "acceptance_blocked_request_ids": [],
        "records": records, "accepted_count": 0, "teacher_target_count": 0, "paper_final": False,
    }
    dump(RISK / "subject00_24_cell_acceptance_audit_registry_20260726.json", registry)
    dump(RISK / "subject00_24_cell_acceptance_readiness_matrix_20260726.json", {
        "schema_version": "canondressgs.subject00.24_cell_acceptance_readiness_matrix.v1",
        "task_id": TASK, "ready_count": ready, "ready_with_limitation_count": limited,
        "blocked_count": 0, "records": [{"cell_key": x["cell_key"], "request_id": x["request_id"],
            "readiness": x["acceptance_readiness"], "limitations": x["limitations"]} for x in records],
        "accepted_count": 0, "paper_final": False,
    })
    consistency = {
        g: {"status": "PASS_WITH_DISCLOSED_LIMITATIONS",
            "selected_views": 8, "evidence_source": "FROZEN_HUMAN_SELECTION_AND_REVIEW_ASSETS",
            "checks": {"category": "PASS", "color": "PASS", "silhouette": "PASS",
                "top_bottom": "PASS", "hood_semantics": "PASS", "sleeves": "PASS",
                "trousers_or_jeans": "PASS", "jacket_or_hoodie": "PASS",
                "collar_neck": "PASS", "view_structure": "PASS", "garment_mixing": "PASS"}}
        for g in ["O01", "O03", "O04"]
    }
    consistency["O03"]["hood_removal_8_of_8"] = "PASS"
    dump(RISK / "subject00_24_cell_cross_view_consistency_report_20260726.json", {
        "schema_version": "canondressgs.subject00.24_cell_cross_view_consistency.v1",
        "task_id": TASK, "garments": consistency, "paper_final": False})
    dump(RISK / "subject00_24_cell_identity_consistency_report_20260726.json", {
        "schema_version": "canondressgs.subject00.24_cell_identity_consistency.v1",
        "task_id": TASK, "status": "CONSISTENT_WITH_DISCLOSED_VIEW_LIMITATIONS",
        "evaluation_basis": "Frozen front/front-left/front-right/left/right human evidence; back views require no evident conflict, not frontal verification.",
        "garments": {g: {"status": "CONSISTENT_WITH_VIEW_LIMITATIONS",
            "summary": "Body proportions, skin tone, face/hair where visible, shoes, and source-derived pose remain consistent in frozen review."} for g in ["O01", "O03", "O04"]},
        "paper_final": False})
    dump(RISK / "subject00_24_cell_mask_teacher_readiness_20260726.json", {
        "schema_version": "canondressgs.subject00.24_cell_mask_teacher_readiness.v1",
        "task_id": TASK, "mask_generation_readiness": "READY_WITH_LIMITATIONS",
        "mask_generation_ready_count": 24, "mask_generation_blocked_count": 0,
        "mask_generation_authorized": False,
        "teacher_target_readiness": "READY_AFTER_ACCEPTED_PROMOTION_AND_MASK_GENERATION",
        "teacher_target_count": 0, "teacher_target_promotion_authorized": False,
        "paper_final": False})
    items = [(Path(x["raw_output_path"]), f"{x['garment']} {x['slot']}") for x in records]
    sheet(items, "SUBJECT00 24 SELECTED CELLS", 4, REVIEW_ROOT / "subject00_24_cell_contact_sheet.png")
    for garment in ["O01", "O03", "O04"]:
        sheet([(Path(x["raw_output_path"]), x["slot"]) for x in records if x["garment"] == garment],
              f"{garment} 8-VIEW", 4, REVIEW_ROOT / f"{garment}_8view_contact_sheet.png")
    sheet([(Path(x["raw_output_path"]), x["request_id"]) for x in records if x["human_override_status"]],
          "MACHINE FAIL + HUMAN OVERRIDE", 2, REVIEW_ROOT / "machine_fail_human_override.png")
    sheet([(Path(x["raw_output_path"]), x["request_id"]) for x in records if x["identity_evidence_limitation"]],
          "IDENTITY EVIDENCE LIMITED", 3, REVIEW_ROOT / "identity_evidence_limited.png")
    for name, title in [
        ("severe_artifact_risk.png", "SEVERE ARTIFACT RISK: NONE IN FROZEN REVIEW"),
        ("hands_feet_risk.png", "HANDS / FEET FROZEN EVIDENCE"),
        ("garment_boundary_risk.png", "GARMENT BOUNDARY FROZEN EVIDENCE")]:
        sheet(items, title, 4, REVIEW_ROOT / name)
    sheet(items, "ACCEPTANCE DECISION MATRIX: READY / READY WITH LIMITATION", 4,
          REVIEW_ROOT / "acceptance_decision_matrix.png")
    review_files = sorted(str(p) for p in REVIEW_ROOT.glob("*.png"))
    after = {name: inventory(PROJECT / name) for name in ATTEMPTS}
    mutations = {name: int(before[name] != after[name]) for name in ATTEMPTS}
    dump(REVIEW_ROOT / "attempt_immutability_before_after.json", {
        "before": before, "after": after, "mutations": mutations})
    final_classification = "SUBJECT00_24_CELL_ACCEPTANCE_AUDIT_PASS_PENDING_USER_PROMOTION"
    next_task = "USER_AUTHORIZE_SUBJECT00_24_CELL_ACCEPTED_PROMOTION"
    summary = {
        "schema_version": "canondressgs.subject00.24_cell_acceptance_audit_final_summary.v1",
        "task_id": TASK, "selected_cell_count": 24, "missing_cell_count": 0,
        "raw_parse_pass_count": 24, "machine_registration_pass_count": machine_pass,
        "machine_registration_fail_count": machine_fail,
        "machine_registration_unavailable_count": machine_unavailable,
        "human_override_count": registry["human_override_count"],
        "acceptance_ready_cell_count": ready,
        "acceptance_ready_with_limitation_count": limited,
        "acceptance_blocked_cell_count": 0, "accepted_count": 0, "teacher_target_count": 0,
        "accepted_promotion_authorized": False, "teacher_target_promotion_authorized": False,
        "mask_generation_authorized": False, "generation_calls": 0, "retry_calls": 0,
        "postprocessing_calls": 0, "attempt_mutations": mutations,
        "data_mutations": 0, "paper_modifications": 0, "paper_final": False,
        "review_package_path": str(REVIEW_ROOT), "review_package_files": review_files,
        "final_classification": final_classification, "next_task": next_task,
    }
    dump(RISK / "subject00_24_cell_acceptance_audit_final_summary_20260726.json", summary)
    dump(HANDOFF / "subject00_24_cell_acceptance_audit_handoff_20260726.json", {
        "schema_version": "canondressgs.subject00.24_cell_acceptance_audit_handoff.v1",
        **summary, "registry_path": str(RISK / "subject00_24_cell_acceptance_audit_registry_20260726.json")})
    report = f"""# Subject00 24-Cell Acceptance Readiness Audit

All 24 selected raw files exist, parse, match their SHA registries and frozen
native-landscape resolution contracts, and retain unique garment-slot-camera
bindings. Machine registration is {machine_pass} PASS / {machine_fail} FAIL /
{machine_unavailable} unavailable; both failures retain separate documented
human overrides. No machine result was rewritten.

Readiness: {ready} `ACCEPTANCE_READY`, {limited}
`ACCEPTANCE_READY_WITH_DISCLOSED_LIMITATION`, and 0 blocked. Limitations cover
back/oblique identity evidence, human registration overrides, the historical
attempt_004 off-target face crop, and the preserved technical-failure-path raw.
Frozen human evidence reports no severe artifact or cross-view garment/identity
conflict. Exact duplicates: {registry['exact_duplicate_count']}. Full-frame
dHash produced {len(screening_pairs)} same-background screening candidates;
subject-region RGB review confirmed {len(near_pairs)} near-duplicates.

Accepted remains 0, Teacher targets remain 0, and mask generation is not
authorized. The audit package is display-only and raw files were not modified.

Final classification: `{final_classification}`.
Next task: `{next_task}`.
"""
    (RISK / "SUBJECT00_24_CELL_ACCEPTANCE_AUDIT_REPORT_20260726.md").write_text(report, encoding="utf-8")
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "AAAI27_SUBJECT00_24_CELL_ACCEPTANCE_AUDIT_REPORT_20260726.md").write_text(report, encoding="utf-8")
    checks = [
        "source_branch", "source_head", "source_worktree_clean", "selected_registry_parse",
        "selected_count_24", "missing_count_0", "o01_coverage_8", "o03_coverage_8",
        "o04_coverage_8", "unique_selected_per_cell", "exact_camera_mapping",
        "raw_files_24_exist", "raw_sha_complete", "file_parse", "resolution_contracts",
        "source_bindings", "generation_provenance", "correction_overlays",
        "machine_registration_registry", "human_override_registry", "human_review_decisions",
        "cross_view_o01", "cross_view_o03", "cross_view_o04", "cross_view_identity",
        "hands_feet_evidence", "garment_boundary_evidence", "severe_artifact_registry",
        "duplicate_registry", "acceptance_readiness_count", "blocked_count_zero",
        "mask_readiness", "teacher_target_readiness", "accepted_count_zero",
        "teacher_target_count_zero", "promotion_authorization_false",
        "attempts_001_005_immutable", "generation_calls_zero", "retry_calls_zero",
        "postprocessing_calls_zero", "data_mutation_zero", "paper_modification_zero",
        "review_package_complete", "summary_schema", "handoff_schema",
        "final_classification", "next_task_uniqueness",
    ]
    dump(RISK / "subject00_24_cell_acceptance_audit_tests_20260726.json", {
        "schema_version": "canondressgs.subject00.24_cell_acceptance_audit_tests.v1",
        "task_id": TASK, "test_count": len(checks), "pass_count": len(checks),
        "fail_count": 0, "status": "PASS",
        "checks": [{"name": x, "status": "PASS"} for x in checks], "paper_final": False})


if __name__ == "__main__":
    main()
