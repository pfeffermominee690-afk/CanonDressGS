#!/usr/bin/env python3
"""Build the evidence-only Subject00 portrait canary visual-fail adjudication."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF = ROOT / "project_control_handoff"
ATTEMPT_001 = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_001")
ATTEMPT_002 = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_002_portrait_canary")
CONTACT_SHEET = ATTEMPT_002 / "05_human_review" / "subject00_portrait_canary_contact_sheet.png"
ORIGINAL_REVIEW_MANIFEST = ATTEMPT_002 / "05_human_review" / "portrait_canary_review_manifest.json"

TASK_ID = "AAAI27-SUBJECT00-PORTRAIT-CANARY-VISUAL-FAIL-ADJUDICATION-001"
SOURCE_BRANCH = "research/subject00-managed-portrait-canary-20260725"
SOURCE_HEAD = "4a1eeb2e53fc1a8b5a5bf1c3b706aac374e156dc"
PORTRAIT_CANARY_RESULT_HEAD = "9742c675943c2b105a5b153888e5b160b73471f2"
BRANCH = "research/subject00-portrait-canary-visual-fail-adjudication-20260726"
REVIEWER = "USER_AND_GPT_MANUAL_REVIEW"
REVIEW_TIMESTAMP = "2026-07-26T07:05:51+08:00"
CLASSIFICATION = "SUBJECT00_PORTRAIT_CANARY_TECHNICAL_PASS_VISUAL_FAIL"
FINAL_CLASSIFICATION = "SUBJECT00_PORTRAIT_CANARY_VISUAL_FAIL_FORMALLY_RECORDED_NEXT_PROTOCOL_PENDING_USER_DECISION"
NEXT_TASK = "USER_SELECT_SUBJECT00_PORTRAIT_CANARY_V2_PROTOCOL"
FORMAL_BASE = "PENDING"
PAPER_FINAL = False

MANUAL_OBSERVATIONS = {
    "subject00_O01_slot00_cand00": {
        "positive_observations": [
            "single person",
            "O01 light hoodie is basically correct",
            "limbs are broadly complete",
            "front direction is broadly preserved",
        ],
        "failure_reasons": [
            "subject is visibly smaller than the original condition",
            "large gray blank regions remain above and below the scene",
            "original camera framing is not preserved",
            "subject pixel size is insufficient",
            "face and identity details are too small for reliable review",
            "not suitable as a registration Teacher target",
        ],
    },
    "subject00_O03_slot03_cand00": {
        "positive_observations": [
            "single person",
            "O03 dark suit is basically correct",
            "side direction is broadly preserved",
            "no obvious duplicate person or severe limb anomaly",
        ],
        "failure_reasons": [
            "subject scale is visibly reduced",
            "large gray blank regions remain above and below the scene",
            "apparent camera distance is increased",
            "framing is inconsistent with the original condition",
            "identity and garment-boundary detail is insufficient",
            "not suitable as a registration Teacher target",
        ],
    },
    "subject00_O04_slot02_cand00": {
        "positive_observations": [
            "single person",
            "O04 dark jacket and jeans are basically correct",
            "the subject is broadly complete",
        ],
        "failure_reasons": [
            "subject is visibly reduced in scale",
            "the scene is embedded in a portrait gray canvas",
            "the lower scene-to-gray transition is unnatural",
            "camera framing does not match the original condition",
            "garment-boundary and identity detail is insufficient",
            "not suitable as a registration Teacher target",
        ],
    },
    "subject00_O01_slot06_cand00": {
        "positive_observations": [
            "single person",
            "O01 light hoodie is basically correct",
            "back direction is broadly preserved",
            "no obvious duplicate person or severe limb anomaly",
        ],
        "failure_reasons": [
            "subject is visibly reduced in scale",
            "large gray regions remain above and below the scene",
            "the original close back-view composition becomes a distant composition",
            "camera registration and subject scale are not preserved",
            "subject detail is insufficient",
            "not suitable as a registration Teacher target",
        ],
    },
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
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_doc(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8", newline="\n")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def file_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def full_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def main() -> int:
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("wrong adjudication branch")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT
    ).returncode
    if ancestor != 0:
        raise RuntimeError("source HEAD is not an ancestor of the adjudication branch")
    allowed_untracked = {
        "?? tools/datasets/build_subject00_portrait_canary_visual_fail_adjudication.py",
        "?? tools/datasets/check_subject00_portrait_canary_visual_fail_adjudication.py",
    }
    unexpected = set(filter(None, git("status", "--short").splitlines())) - allowed_untracked
    if unexpected:
        raise RuntimeError(f"unexpected pre-build worktree changes: {sorted(unexpected)}")

    external_contract = load_json(ATTEMPT_002 / "00_contract" / "subject00_portrait_canary_contract.json")
    canvas_registry = load_json(RISK / "subject00_portrait_canvas_registry.json")
    output_registry = load_json(RISK / "subject00_portrait_canary_output_registry.json")
    original_manifest = load_json(ORIGINAL_REVIEW_MANIFEST)
    if any(item.get(field) is not None for item in original_manifest["records"] for field in [
        "identity_match", "pose_match", "camera_match", "garment_match", "full_body_complete",
        "hands_complete", "feet_complete", "background_match", "artifact_grade", "decision",
    ]):
        raise RuntimeError("original review manifest no longer has null visual fields")

    attempt_001_records = []
    for item in external_contract["attempt_001_baseline"]:
        path = Path(item["path"])
        observed = file_sha256(path)
        if observed != item["sha256"]:
            raise RuntimeError(f"attempt_001 mutation: {item['request_id']}")
        attempt_001_records.append({**item, "bytes": path.stat().st_size, "observed_sha256": observed})

    attempt_002_inventory = full_inventory(ATTEMPT_002)
    contact_record = file_record(CONTACT_SHEET, "HUMAN_REVIEW_CONTACT_SHEET")
    manifest_record = file_record(ORIGINAL_REVIEW_MANIFEST, "ORIGINAL_NULL_REVIEW_MANIFEST")
    canvas_by_id = {item["request_id"]: item for item in canvas_registry["records"]}
    output_by_id = {item["request_id"]: item for item in output_registry["records"]}
    request_ids = list(MANUAL_OBSERVATIONS)

    with Image.open(CONTACT_SHEET) as image:
        image.load()
        contact_record.update({"format": image.format, "width": image.width, "height": image.height})

    review_records = []
    for request_id in request_ids:
        canvas = canvas_by_id[request_id]
        output = output_by_id[request_id]
        output_path = Path(output["output_path"])
        with Image.open(output_path) as image:
            image.load()
            png_parse_pass = image.format == "PNG"
            native_resolution_pass = image.size == (1024, 1536)
        review_records.append(
            {
                "request_id": request_id,
                "garment": output["garment_id"],
                "slot": output["slot_id"],
                "camera": output["camera_id"],
                "original_condition_path": canvas["source_path"],
                "original_condition_sha256": canvas["source_sha256"],
                "derived_canvas_path": canvas["derived_canvas_path"],
                "derived_canvas_sha256": canvas["derived_canvas_sha256"],
                "generated_output_path": output["output_path"],
                "generated_output_sha256": output["output_sha256"],
                "native_resolution_pass": native_resolution_pass,
                "png_parse_pass": png_parse_pass,
                "single_person": True,
                "garment_correctness": "BASICALLY_CORRECT",
                "pose_direction_preservation": "BROADLY_PRESERVED",
                "subject_scale_preservation": "FAIL",
                "camera_framing_preservation": "FAIL",
                "background_extension_quality": "FAIL",
                "identity_reviewability": "FAIL",
                "garment_boundary_reviewability": "FAIL",
                "human_visual_decision": "FAIL",
                "positive_observations": MANUAL_OBSERVATIONS[request_id]["positive_observations"],
                "failure_reasons": MANUAL_OBSERVATIONS[request_id]["failure_reasons"],
                "accepted": False,
                "teacher_target": False,
                "reviewer": REVIEWER,
                "review_timestamp": REVIEW_TIMESTAMP,
                "overall_classification": CLASSIFICATION,
            }
        )

    evidence = {
        "schema_version": "canondressgs.subject00.portrait_canary_visual_fail_evidence.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "portrait_canary_result_head": PORTRAIT_CANARY_RESULT_HEAD,
        "source_worktree_clean_at_gate": True,
        "contact_sheet": contact_record,
        "contact_sheet_group_count_confirmed_by_manual_open": 4,
        "original_review_manifest": manifest_record,
        "attempt_001_output_count": len(attempt_001_records),
        "attempt_001_outputs": attempt_001_records,
        "attempt_001_preserved_native_pass_outputs": [
            item for item in attempt_001_records if item["preserved_native_pass_output"]
        ],
        "attempt_002_file_count": len(attempt_002_inventory),
        "attempt_002_files": attempt_002_inventory,
        "derived_canvases": [file_record(Path(item["derived_canvas_path"]), "DERIVED_EDIT_INPUT_CANVAS") for item in canvas_registry["records"]],
        "generated_outputs": [file_record(Path(item["output_path"]), "PORTRAIT_CANARY_GENERATED_OUTPUT") for item in output_registry["records"]],
        "new_generation_calls": 0,
        "external_api_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
        "paper_final": PAPER_FINAL,
    }
    adjudication = {
        "schema_version": "canondressgs.subject00.portrait_canary_manual_visual_adjudication.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "portrait_canary_result_head": PORTRAIT_CANARY_RESULT_HEAD,
        "final_reporting_head": SOURCE_HEAD,
        "contact_sheet_path": str(CONTACT_SHEET),
        "contact_sheet_sha256": contact_record["sha256"],
        "original_review_manifest_path": str(ORIGINAL_REVIEW_MANIFEST),
        "original_review_manifest_sha256": manifest_record["sha256"],
        "request_count": 4,
        "human_visual_pass_count": 0,
        "human_visual_fail_count": 4,
        "records": review_records,
        "common_failure_causes": [
            "subject pixel height is substantially reduced",
            "camera framing changes and apparent subject distance increases",
            "large gray canvas regions remain instead of continuous scene extension",
            "identity and garment-boundary reviewability are reduced",
            "the result is incompatible with fixed pose/camera registration targets",
        ],
        "reviewer": REVIEWER,
        "review_timestamp": REVIEW_TIMESTAMP,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "formal_base": FORMAL_BASE,
        "remaining_39_rerun_authorization": "DENIED",
        "subject00_paper_positive_claim_count": 0,
        "overall_classification": CLASSIFICATION,
        "paper_final": PAPER_FINAL,
    }
    overlay = {
        "schema_version": "canondressgs.subject00.portrait_canary_review_overlay.v1",
        "task_id": TASK_ID,
        "overlay_only": True,
        "original_manifest_path": str(ORIGINAL_REVIEW_MANIFEST),
        "original_manifest_sha256": manifest_record["sha256"],
        "original_manifest_modified": False,
        "reviewer": REVIEWER,
        "review_timestamp": REVIEW_TIMESTAMP,
        "records": [
            {
                "request_id": item["request_id"],
                "human_visual_decision": item["human_visual_decision"],
                "failure_reasons": item["failure_reasons"],
                "accepted": item["accepted"],
                "teacher_target": item["teacher_target"],
            }
            for item in review_records
        ],
        "paper_final": PAPER_FINAL,
    }
    summary = {
        "schema_version": "canondressgs.subject00.portrait_canary_visual_fail_final_summary.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "portrait_canary_result_head": PORTRAIT_CANARY_RESULT_HEAD,
        "branch": BRANCH,
        "engineering_pass": True,
        "native_resolution_pass_count": 4,
        "human_visual_pass_count": 0,
        "human_visual_fail_count": 4,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "remaining_39_generated_count": 0,
        "remaining_39_rerun_authorized": False,
        "formal_base": FORMAL_BASE,
        "new_generation_calls": 0,
        "external_api_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
        "paper_modifications": 0,
        "subject00_paper_positive_claim_count": 0,
        "tests": "PENDING",
        "classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "next_task_authorized": False,
        "paper_final": PAPER_FINAL,
    }
    handoff = {
        "schema_version": "canondressgs.subject00.portrait_canary_visual_fail_adjudication_handoff.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "portrait_canary_result_head": PORTRAIT_CANARY_RESULT_HEAD,
        "branch": BRANCH,
        "classification": FINAL_CLASSIFICATION,
        "engineering_status": "PASS",
        "native_resolution_status": "PASS_4_OF_4",
        "visual_status": "FAIL_4_OF_4",
        "accepted_count": 0,
        "teacher_target_count": 0,
        "formal_base": FORMAL_BASE,
        "remaining_39_rerun_authorized": False,
        "tests": "PENDING",
        "next_task": NEXT_TASK,
        "next_task_authorized": False,
        "paper_final": PAPER_FINAL,
    }

    write_json(RISK / "subject00_portrait_canary_visual_fail_evidence_registry_20260726.json", evidence)
    write_json(RISK / "subject00_portrait_canary_manual_visual_adjudication_20260726.json", adjudication)
    write_json(RISK / "subject00_portrait_canary_review_overlay_20260726.json", overlay)
    write_json(RISK / "subject00_portrait_canary_visual_fail_final_summary_20260726.json", summary)
    write_json(HANDOFF / "subject00_portrait_canary_visual_fail_adjudication_handoff.json", handoff)

    write_doc(
        RISK / "SUBJECT00_PORTRAIT_CANARY_VISUAL_FAIL_REPORT_20260726.md",
        f"""
# Subject00 Portrait Canary Visual Fail Report

- Task: `{TASK_ID}`
- Source: `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`
- Portrait canary result: `{PORTRAIT_CANARY_RESULT_HEAD}`
- Engineering result: `PASS`
- Native-resolution result: `PASS 4/4`
- Human visual result: `FAIL 4/4`
- Accepted / Teacher targets: `0 / 0`
- Formal Base: `PENDING`
- Remaining 39 authorization: `DENIED`
- PAPER_FINAL: `false`

The fixed user review rejects all four canaries. The portrait canvas contract returned native 1024x1536 PNG files, but the contained 1330x1150 condition was reduced to 960x830 at offset (32,353). The model retained the appearance of a reduced landscape scene embedded in a gray portrait canvas instead of preserving subject scale and camera registration while extending the scene.

The common visual failures are reduced subject pixel height, changed camera framing, increased apparent camera distance, large gray canvas areas, weakened identity reviewability, and weakened garment-boundary reviewability. Therefore `NATIVE_RESOLUTION_PASS` is not `CAMERA_REGISTRATION_PASS` and is not `FORMAL_DATASET_PASS`.

The historical technical result remains valid as engineering evidence. This adjudication supersedes its pending visual status with `{CLASSIFICATION}`. No image is accepted, no Teacher target is created, and no additional generation is authorized.
""",
    )
    write_doc(
        RISK / "SUBJECT00_RESOLUTION_CAMERA_REGISTRATION_CONTRACT_ADJUDICATION_20260726.md",
        f"""
# Subject00 Resolution and Camera Registration Contract Adjudication

## Proven Facts

- Managed Image Edit can return native 1024x1536 PNG from a portrait derived canvas.
- The current contain-plus-gray-padding construction substantially reduces subject pixel scale.
- The native-resolution contract passes independently.
- The camera framing, subject scale, and registration contract fails.
- All four canaries remain failed evidence and must not be replaced or hidden.

## Mandatory V2 Gates

Every V2 candidate must return native 1024x1536 without output resize, crop, pad, rotate, stretch, re-encoding repair, or canvas embedding. The complete person and pose/camera direction must remain stable. Subject pixel scale must not be materially lower than the condition. The image must not contain large pure-gray or visibly artificial canvas regions. Background extension must be continuous with the source scene, and identity, hands, feet, and garment boundaries must remain reviewable. Only four canaries may be tested; the remaining 39 stay denied until 4/4 human Visual PASS.

## Candidate Protocols For User Selection

| Candidate | Input canvas construction | Subject size changed | Condition cropped | Camera framing changed | Model extends background | Identity drift risk | Pose drift risk | Native-resolution risk | Scientific-contract fit | Recommendation |
|---|---|---:|---:|---:|---:|---|---|---|---|---|
| A: portrait outpainting / scene extension | Keep the original condition at native subject scale and expand only the vertical scene context to a 1024x1536 edit target | No intended reduction | No | No intended change | Yes | Medium | Medium | Low to medium | Potentially compatible if registration gates pass | HIGH for controlled four-image evaluation |
| B: subject-scale-preserving derived canvas | Scale the source only enough to fit width while preserving subject pixel height; allocate extension outside the source scene without gray filler | Minimal | No | Low intended change | Yes | Medium | Low to medium | Low | Compatible in principle; requires measurable scale tolerance | HIGH for controlled four-image evaluation |
| C: original condition reference plus independent portrait scene target | Image 1 is a portrait scene target with registered camera geometry; Image 2 is the unchanged condition authority | No intended reduction | No | Depends on target registration quality | Yes | Medium to high | Medium | Low | Conditional; target geometry must be independently validated | MEDIUM |
| D: direct portrait recomposition from the condition | Ask the model to synthesize a portrait composition without a deterministic registered scene target | Uncontrolled | No deterministic crop | Likely | Yes | High | High | Medium | Weak registration guarantee | LOW |

These are design candidates only. This task does not select or execute a candidate. The sole next action is `{NEXT_TASK}`.
""",
    )
    print(json.dumps({"status": "BUILT", "request_count": 4, "visual_fail_count": 4}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
