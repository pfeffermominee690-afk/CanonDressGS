#!/usr/bin/env python3
"""Manage Subject00 O03 frozen-contract hood-removal canary execution.

This script does not call any image-generation API. Codex invokes the
platform-managed imagegen tool directly, then this script records and validates
the returned PNGs without resizing, cropping, padding, or re-encoding them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


TASK_ID = "AAAI27-SUBJECT00-O03-FROZEN-CONTRACT-CANARY-EXECUTION-001"
SOURCE_TASK_ID = "AAAI27-SUBJECT00-1349-HUMAN-SELECTION-AND-O03-CANARY-PREP-001"
SOURCE_BRANCH = "research/subject00-1349-human-selection-o03-canary-prep-20260726"
SOURCE_HEAD = "4f4ee407c0ddfde2f38fda149bbbae4797e67d15"
EXECUTION_BRANCH = "research/subject00-o03-frozen-contract-canary-execution-20260726"
ATTEMPT_NAMESPACE = "attempt_004_o03_hood_removal_targeted_canary"
AUTHORIZATION_SCOPE = "O03_FROZEN_TARGETED_CANARY_4_REQUESTS_ONLY"
EXPECTED_SIZE = (1349, 1166)
EXPECTED_REQUEST_IDS = [
    "subject00_O03_slot00_canary_attempt004_cand00",
    "subject00_O03_slot04_canary_attempt004_cand00",
    "subject00_O03_slot05_canary_attempt004_cand00",
    "subject00_O03_slot07_canary_attempt004_cand00",
]
DATA_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001")
ATTEMPT_ROOT = DATA_ROOT / ATTEMPT_NAMESPACE
FROZEN_MANIFEST_REL = Path("paper_protocol/reviewer_risk/subject00_o03_hood_removal_canary_manifest_draft_20260726.json")
FROZEN_CONTRACT_REL = Path("paper_protocol/reviewer_risk/SUBJECT00_O03_HOOD_REMOVAL_CANARY_CONTRACT_20260726.md")
FINAL_SUMMARY_REL = Path("paper_protocol/reviewer_risk/subject00_o03_frozen_contract_canary_final_summary_20260726.json")
EXECUTION_MANIFEST_REL = Path("paper_protocol/reviewer_risk/subject00_o03_frozen_contract_canary_execution_manifest_20260726.json")
PROVENANCE_REGISTRY_REL = Path("paper_protocol/reviewer_risk/subject00_o03_frozen_contract_canary_provenance_registry_20260726.json")
REVIEW_MANIFEST_REL = Path("paper_protocol/reviewer_risk/subject00_o03_frozen_contract_canary_human_review_manifest_20260726.json")
TECHNICAL_REPORT_REL = Path("paper_protocol/reviewer_risk/SUBJECT00_O03_FROZEN_CONTRACT_CANARY_TECHNICAL_REPORT_20260726.md")
HANDOFF_REL = Path("project_control_handoff/subject00_o03_frozen_contract_canary_execution_handoff_20260726.json")

PROVIDER = "CODEX_IMAGE_GENERATION_SKILL"
MODE = "CODEX_PLATFORM_MANAGED_DIRECT_IMAGE_EDIT"
SKILL_NAME = "imagegen"
TOOL_NAME = "image_gen.imagegen"
BACKEND_MODEL_STATUS = "NOT_EXPOSED_BY_PLATFORM"

OLD_ATTEMPTS = {
    "attempt_001": DATA_ROOT / "attempt_001",
    "attempt_002": DATA_ROOT / "attempt_002_portrait_canary",
    "attempt_003": DATA_ROOT / "attempt_003_portrait_canary_v2_registered_outpaint",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value.rstrip() + "\n", encoding="utf-8")
    tmp.replace(path)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8").strip()


def image_meta(path: Path, require_expected: bool = False) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        meta = {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    if require_expected and (meta["width"], meta["height"]) != EXPECTED_SIZE:
        raise ValueError(f"Expected {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}, got {meta['width']}x{meta['height']}")
    if require_expected and meta["format"] != "PNG":
        raise ValueError(f"Expected PNG, got {meta['format']}")
    if require_expected and meta["size_bytes"] <= 0:
        raise ValueError("Output bytes must be > 0")
    return meta


def load_frozen_manifest(repo: Path) -> dict[str, Any]:
    manifest = read_json(repo / FROZEN_MANIFEST_REL)
    if manifest["task_id"] != SOURCE_TASK_ID:
        raise RuntimeError("Frozen manifest task_id mismatch")
    if manifest["attempt_namespace"] != ATTEMPT_NAMESPACE:
        raise RuntimeError("Frozen manifest attempt namespace mismatch")
    if manifest["request_count"] != 4:
        raise RuntimeError("Frozen manifest request_count mismatch")
    if [r["request_id"] for r in manifest["requests"]] != EXPECTED_REQUEST_IDS:
        raise RuntimeError("Frozen manifest request IDs mismatch")
    size = manifest["expected_resolution"]
    if (size["width"], size["height"]) != EXPECTED_SIZE:
        raise RuntimeError("Frozen manifest expected resolution mismatch")
    retry = manifest["retry_policy"]
    if retry["automatic_retry"] or retry["retry_count"] != 0 or retry["candidate_count_per_cell"] != 1:
        raise RuntimeError("Frozen manifest retry policy mismatch")
    if any(bool(value) for value in manifest["postprocessing_policy"].values()):
        raise RuntimeError("Frozen manifest postprocessing policy mismatch")
    return manifest


def load_request_records(repo: Path) -> list[dict[str, Any]]:
    manifest = load_frozen_manifest(repo)
    records = []
    for index, item in enumerate(manifest["requests"]):
        source_path = Path(item["local_source_condition_path"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_sha = sha256(source_path)
        if source_sha != item["source_condition_sha256"]:
            raise RuntimeError(f"Source condition SHA mismatch: {source_path}")
        source_request_path = Path(item["source_request_path"])
        if not source_request_path.is_file():
            raise FileNotFoundError(source_request_path)
        if sha256(source_request_path) != item["source_request_sha256"]:
            raise RuntimeError(f"Source request SHA mismatch: {source_request_path}")
        source_request = read_json(source_request_path)
        prompt = build_prompt(manifest, item, source_request)
        prompt_path = ATTEMPT_ROOT / "02_prompts" / f"{item['request_id']}.txt"
        output_path = ATTEMPT_ROOT / "04_generation_responses" / "codex_managed_candidates" / "O03" / f"{item['request_id']}.png"
        record_path = ATTEMPT_ROOT / "06_provenance" / item["request_id"] / "generation_record.json"
        records.append(
            {
                "index": index,
                "request_id": item["request_id"],
                "source_request_id": item["source_request_id"],
                "garment": item["garment"],
                "slot": item["slot"],
                "camera": item["camera"],
                "direction": item["orientation"],
                "source_condition_path": str(source_path),
                "source_condition_sha256": item["source_condition_sha256"],
                "source_condition_metadata": image_meta(source_path),
                "source_request_path": str(source_request_path),
                "source_request_sha256": item["source_request_sha256"],
                "generation_method": {
                    "generation_provider": PROVIDER,
                    "generation_mode": MODE,
                    "source_generation_backend": source_request.get("generation_backend"),
                    "source_generation_mode": source_request.get("generation_mode"),
                    "skill_name": SKILL_NAME,
                    "tool_name": TOOL_NAME,
                    "external_api_used": False,
                    "api_key_used": False,
                    "backend_model": None,
                    "backend_model_status": BACKEND_MODEL_STATUS,
                },
                "prompt_path": str(prompt_path),
                "prompt_sha256": sha256_text(prompt),
                "prompt": prompt,
                "output_path": str(output_path),
                "output_relative_path": str(output_path.relative_to(ATTEMPT_ROOT)).replace("\\", "/"),
                "generation_record_path": str(record_path),
                "expected_resolution": {"width": EXPECTED_SIZE[0], "height": EXPECTED_SIZE[1]},
                "retry_allowed": 0,
                "postprocessing_allowed": False,
                "status": "PENDING",
                "generation_calls": 0,
                "retry_count": 0,
            }
        )
    return records


def build_prompt(manifest: dict[str, Any], item: dict[str, Any], source_request: dict[str, Any]) -> str:
    bullets = "\n".join(f"- {line}" for line in manifest["prompt_contract"])
    return (
        "Use case: identity-preserve\n"
        f"Asset type: Subject00 O03 frozen targeted hood-removal canary {item['request_id']}\n"
        "Input images: Image 1 is the only identity, body, pose, camera, framing, background, and lighting authority.\n"
        f"Source binding: garment O03, slot {item['slot']}, camera {item['camera']}, direction {item['orientation']}.\n"
        "Primary request: Edit clothing only in Image 1 according to the frozen prompt contract below.\n"
        "Frozen prompt contract:\n"
        f"{bullets}\n"
        "Execution invariants:\n"
        f"- Preserve exactly the Image 1 {item['orientation']} view and registered Subject00 pose/camera.\n"
        "- Keep one full-body person from head through both feet; preserve hands, feet, face, hair, ears, neck, lighting, and background.\n"
        "- Remove all blue-and-white hoodie fabric and hood structure from head, neck, shoulders, torso, sleeves, and waist.\n"
        "- Final outfit must be the frozen O03 formal suit: dark navy tailored single-breasted suit jacket, white dress shirt, dark navy tie, and full-length matching trousers.\n"
        f"- Output must be a native PNG at exactly {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]} landscape pixels.\n"
        "- Do not zoom, crop, rotate, outpaint, resize, pad, recompose, re-encode, repair, beautify, change hairstyle, or postprocess.\n"
        "- No text, labels, numbers, watermark, border, collage, extra person, extra limbs, missing hand, missing foot, or changed background.\n"
        f"Source request provenance: {source_request.get('request_id')} / {source_request.get('generation_backend')}.\n"
    )


def tree_snapshot(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"root": str(root), "exists": False, "file_count": 0, "total_bytes": 0, "tree_sha256": None, "files": []}
    files = []
    digest = hashlib.sha256()
    total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_sha = sha256(path)
        total += size
        files.append({"path": rel, "size_bytes": size, "sha256": file_sha})
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
    return {
        "root": str(root),
        "exists": True,
        "file_count": len(files),
        "total_bytes": total,
        "tree_sha256": digest.hexdigest(),
        "files": files,
    }


def old_attempt_snapshots() -> dict[str, Any]:
    return {name: tree_snapshot(path) for name, path in OLD_ATTEMPTS.items()}


def compare_old_attempts(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name in OLD_ATTEMPTS:
        b = before[name]
        a = after[name]
        mutated = b["exists"] != a["exists"] or b["file_count"] != a["file_count"] or b["total_bytes"] != a["total_bytes"] or b["tree_sha256"] != a["tree_sha256"]
        result[name] = {
            "mutations": 1 if mutated else 0,
            "before_file_count": b["file_count"],
            "after_file_count": a["file_count"],
            "before_total_bytes": b["total_bytes"],
            "after_total_bytes": a["total_bytes"],
            "before_tree_sha256": b["tree_sha256"],
            "after_tree_sha256": a["tree_sha256"],
        }
    return result


def prepare(repo: Path) -> dict[str, Any]:
    if ATTEMPT_ROOT.exists():
        raise FileExistsError(f"Attempt root already exists: {ATTEMPT_ROOT}")
    records = load_request_records(repo)
    for subdir in [
        "00_contract",
        "01_source_bindings",
        "02_prompts",
        "03_generation_requests/requests",
        "04_generation_responses/codex_managed_candidates/O03",
        "04_generation_responses/technical_failures",
        "05_review_assets/contact_sheet",
        "05_review_assets/side_by_side",
        "05_review_assets/crops",
        "06_provenance",
        "07_human_review",
        "08_technical_validation",
        "09_final",
    ]:
        (ATTEMPT_ROOT / subdir).mkdir(parents=True, exist_ok=False)
    shutil.copyfile(repo / FROZEN_MANIFEST_REL, ATTEMPT_ROOT / "00_contract" / FROZEN_MANIFEST_REL.name)
    shutil.copyfile(repo / FROZEN_CONTRACT_REL, ATTEMPT_ROOT / "00_contract" / FROZEN_CONTRACT_REL.name)
    for record in records:
        atomic_text(Path(record["prompt_path"]), record["prompt"])
        request_sidecar = {
            "schema_version": "canondressgs.subject00.o03_frozen_contract_canary.request.v1",
            "task_id": TASK_ID,
            **{k: v for k, v in record.items() if k != "prompt"},
        }
        atomic_json(ATTEMPT_ROOT / "03_generation_requests" / "requests" / f"{record['request_id']}.json", request_sidecar)
    before = old_attempt_snapshots()
    manifest = {
        "schema_version": "canondressgs.subject00.o03_frozen_contract_canary.execution_manifest.v1",
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "execution_branch": EXECUTION_BRANCH,
        "attempt_namespace": ATTEMPT_NAMESPACE,
        "attempt_root": str(ATTEMPT_ROOT),
        "authorization_scope": AUTHORIZATION_SCOPE,
        "generation_authorized": True,
        "generation_provider": PROVIDER,
        "generation_mode": MODE,
        "tool_name": TOOL_NAME,
        "external_api_used": False,
        "api_key_used": False,
        "expected_resolution": {"width": EXPECTED_SIZE[0], "height": EXPECTED_SIZE[1]},
        "request_count": len(records),
        "generation_calls_max": 4,
        "generation_calls": 0,
        "output_count": 0,
        "png_parse_pass_count": 0,
        "native_resolution_pass_count": 0,
        "duplicate_sha_count": 0,
        "status": "READY_FOR_GENERATION",
        "records": records,
    }
    atomic_json(ATTEMPT_ROOT / "01_source_bindings" / "source_condition_bindings.json", {"records": records})
    atomic_json(ATTEMPT_ROOT / "06_provenance" / "old_attempt_snapshot_before.json", before)
    atomic_json(ATTEMPT_ROOT / "06_provenance" / "generation_manifest.json", manifest)
    atomic_json(ATTEMPT_ROOT / "09_final" / "start_state.json", {"created_at": now(), "status": "READY_FOR_GENERATION", "old_attempt_snapshot_before": before})
    return {"status": "READY_FOR_GENERATION", "attempt_root": str(ATTEMPT_ROOT), "request_count": len(records)}


def load_generation_manifest() -> tuple[Path, dict[str, Any]]:
    path = ATTEMPT_ROOT / "06_provenance" / "generation_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, read_json(path)


def next_pending() -> dict[str, Any]:
    _, manifest = load_generation_manifest()
    for record in manifest["records"]:
        if record["status"] == "PENDING":
            return record
    return {"status": "NO_PENDING_REQUESTS"}


def update_counts(manifest: dict[str, Any]) -> None:
    outputs = [record for record in manifest["records"] if record["status"] == "SUCCESS"]
    shas = [record["output"]["sha256"] for record in outputs]
    manifest["output_count"] = len(outputs)
    manifest["png_parse_pass_count"] = sum(1 for record in outputs if record["output"].get("format") == "PNG")
    manifest["native_resolution_pass_count"] = sum(1 for record in outputs if (record["output"].get("width"), record["output"].get("height")) == EXPECTED_SIZE)
    manifest["duplicate_sha_count"] = len(shas) - len(set(shas))


def record_success(request_id: str, tool_output: Path, tool_output_hint: str | None) -> dict[str, Any]:
    manifest_path, manifest = load_generation_manifest()
    matches = [record for record in manifest["records"] if record["request_id"] == request_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown request_id: {request_id}")
    record = matches[0]
    if record["status"] != "PENDING":
        raise RuntimeError(f"Request is not pending: {request_id} / {record['status']}")
    if manifest["generation_calls"] >= 4:
        raise RuntimeError("Generation call budget exhausted")
    meta = image_meta(tool_output, require_expected=True)
    previous_shas = {r.get("output", {}).get("sha256") for r in manifest["records"] if r.get("output")}
    if meta["sha256"] in previous_shas:
        raise RuntimeError(f"Duplicate output SHA: {meta['sha256']}")
    output_path = Path(record["output_path"])
    if output_path.exists():
        raise FileExistsError(output_path)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tool_output, tmp)
    if sha256(tmp) != meta["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Immutable copy SHA mismatch")
    tmp.replace(output_path)
    copied_meta = image_meta(output_path, require_expected=True)
    record["status"] = "SUCCESS"
    record["generation_calls"] = 1
    record["output"] = {"path": str(output_path), **copied_meta}
    record["tool_output_hint"] = tool_output_hint
    record["postprocessing_applied"] = False
    generation_record = {
        "schema_version": "canondressgs.subject00.o03_frozen_contract_canary.generation_record.v1",
        "task_id": TASK_ID,
        "request_id": request_id,
        "created_at": now(),
        "generation_provider": PROVIDER,
        "generation_mode": MODE,
        "skill_name": SKILL_NAME,
        "tool_name": TOOL_NAME,
        "external_api_used": False,
        "api_key_used": False,
        "backend_model": None,
        "backend_model_status": BACKEND_MODEL_STATUS,
        "source_condition_path": record["source_condition_path"],
        "source_condition_sha256": record["source_condition_sha256"],
        "prompt_path": record["prompt_path"],
        "prompt_sha256": record["prompt_sha256"],
        "tool_output_hint": tool_output_hint,
        "output": record["output"],
        "retry_count": 0,
        "postprocessing_applied": False,
        "raw_output_immutable": True,
    }
    atomic_json(Path(record["generation_record_path"]), generation_record)
    manifest["generation_calls"] += 1
    update_counts(manifest)
    manifest["status"] = "COMPLETE_PENDING_FINALIZATION" if manifest["output_count"] == 4 else "PARTIAL_GENERATION_IN_PROGRESS"
    atomic_json(manifest_path, manifest)
    return {"status": "SUCCESS", "request_id": request_id, "output": record["output"], "generation_calls": manifest["generation_calls"]}


def record_failure(request_id: str, error_type: str, tool_output: Path | None = None, detail: str | None = None) -> dict[str, Any]:
    manifest_path, manifest = load_generation_manifest()
    matches = [record for record in manifest["records"] if record["request_id"] == request_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown request_id: {request_id}")
    record = matches[0]
    if record["status"] != "PENDING":
        raise RuntimeError(f"Request is not pending: {request_id} / {record['status']}")
    if manifest["generation_calls"] >= 4:
        raise RuntimeError("Generation call budget exhausted")
    failure: dict[str, Any] = {"error_type": error_type, "detail": detail, "recorded_at": now()}
    if tool_output is not None and tool_output.is_file():
        failure_path = ATTEMPT_ROOT / "04_generation_responses" / "technical_failures" / f"{request_id}.png"
        if not failure_path.exists():
            shutil.copyfile(tool_output, failure_path)
        try:
            failure["tool_output"] = {"path": str(failure_path), **image_meta(failure_path)}
        except Exception as exc:  # noqa: BLE001 - preserve parse failure detail
            failure["tool_output"] = {"path": str(failure_path), "parse_error": str(exc), "size_bytes": failure_path.stat().st_size, "sha256": sha256(failure_path)}
    record["status"] = "TECHNICAL_FAILURE"
    record["generation_calls"] = 1
    record["technical_failure"] = failure
    manifest["generation_calls"] += 1
    manifest["status"] = "PARTIAL_GENERATION_PAUSED"
    update_counts(manifest)
    atomic_json(manifest_path, manifest)
    return {"status": "TECHNICAL_FAILURE", "request_id": request_id, "generation_calls": manifest["generation_calls"], "failure": failure}


def make_contact_sheet(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    tile_w, tile_h, label_h = 675, 583, 34
    cols = 2
    rows = (len(records) + 1) // 2
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, record in enumerate(records):
        with Image.open(record["output"]["path"]) as image:
            tile = image.convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = (index % cols) * tile_w
        y = (index // cols) * (tile_h + label_h)
        canvas.paste(tile, (x, y))
        draw.text((x + 8, y + tile_h + 8), record["request_id"], fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def make_side_by_side(record: dict[str, Any], path: Path) -> None:
    with Image.open(record["source_condition_path"]) as src, Image.open(record["output"]["path"]) as out:
        src_rgb = src.convert("RGB")
        out_rgb = out.convert("RGB")
        height = max(src_rgb.height, out_rgb.height)
        width = src_rgb.width + out_rgb.width
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(src_rgb, (0, 0))
        canvas.paste(out_rgb, (src_rgb.width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def crop_box(width: int, height: int, kind: str) -> tuple[int, int, int, int]:
    boxes = {
        "face_head": (0.34, 0.00, 0.66, 0.27),
        "neck_shoulder": (0.24, 0.11, 0.76, 0.38),
        "full_body": (0.00, 0.00, 1.00, 1.00),
        "hands_feet": (0.08, 0.38, 0.92, 1.00),
        "garment_boundary": (0.18, 0.18, 0.82, 0.88),
    }
    left, top, right, bottom = boxes[kind]
    return (int(width * left), int(height * top), int(width * right), int(height * bottom))


def make_crops(record: dict[str, Any], crop_root: Path) -> dict[str, str]:
    outputs = {}
    with Image.open(record["output"]["path"]) as image:
        rgb = image.convert("RGB")
        for kind in ["face_head", "neck_shoulder", "full_body", "hands_feet", "garment_boundary"]:
            path = crop_root / record["request_id"] / f"{kind}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            rgb.crop(crop_box(rgb.width, rgb.height, kind)).save(path, format="PNG")
            outputs[kind] = str(path)
    return outputs


def finalize(repo: Path) -> dict[str, Any]:
    manifest_path, manifest = load_generation_manifest()
    successes = [record for record in manifest["records"] if record["status"] == "SUCCESS"]
    contact_sheet = ATTEMPT_ROOT / "05_review_assets" / "contact_sheet" / "subject00_o03_frozen_contract_canary_2x2_contact_sheet.png"
    make_contact_sheet(successes, contact_sheet)
    side_by_side_paths = {}
    crop_paths = {}
    for record in successes:
        side_path = ATTEMPT_ROOT / "05_review_assets" / "side_by_side" / f"{record['request_id']}_source_output_side_by_side.png"
        make_side_by_side(record, side_path)
        side_by_side_paths[record["request_id"]] = str(side_path)
        crop_paths[record["request_id"]] = make_crops(record, ATTEMPT_ROOT / "05_review_assets" / "crops")
    after = old_attempt_snapshots()
    before = read_json(ATTEMPT_ROOT / "06_provenance" / "old_attempt_snapshot_before.json")
    old_attempt_comparison = compare_old_attempts(before, after)
    review_records = []
    for record in manifest["records"]:
        output = record.get("output")
        review_records.append(
            {
                "request_id": record["request_id"],
                "garment": record["garment"],
                "slot": record["slot"],
                "camera": record["camera"],
                "direction": record["direction"],
                "source_condition_path": record["source_condition_path"],
                "source_condition_sha256": record["source_condition_sha256"],
                "output_path": output["path"] if output else None,
                "output_sha256": output["sha256"] if output else None,
                "output_resolution": {"width": output["width"], "height": output["height"]} if output else None,
                "technical_status": record["status"],
                "side_by_side_path": side_by_side_paths.get(record["request_id"]),
                "crop_paths": crop_paths.get(record["request_id"], {}),
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
    final_classification = (
        "SUBJECT00_O03_FROZEN_CONTRACT_CANARY_GENERATED_PENDING_HUMAN_REVIEW"
        if manifest["output_count"] == 4
        and manifest["png_parse_pass_count"] == 4
        and manifest["native_resolution_pass_count"] == 4
        and manifest["duplicate_sha_count"] == 0
        and all(item["mutations"] == 0 for item in old_attempt_comparison.values())
        else "SUBJECT00_O03_FROZEN_CONTRACT_CANARY_PARTIAL_PENDING_USER_REVIEW"
    )
    if any(item["mutations"] for item in old_attempt_comparison.values()):
        final_classification = "SUBJECT00_O03_CANARY_CONTRACT_VIOLATION_OLD_ATTEMPT_MUTATION"
    next_task = (
        "USER_REVIEW_SUBJECT00_O03_FROZEN_CONTRACT_CANARY"
        if final_classification == "SUBJECT00_O03_FROZEN_CONTRACT_CANARY_GENERATED_PENDING_HUMAN_REVIEW"
        else "USER_REVIEW_PARTIAL_SUBJECT00_O03_FROZEN_CONTRACT_CANARY"
    )
    review_manifest = {
        "schema_version": "canondressgs.subject00.o03_frozen_contract_canary.human_review_manifest.v1",
        "task_id": TASK_ID,
        "attempt_namespace": ATTEMPT_NAMESPACE,
        "attempt_root": str(ATTEMPT_ROOT),
        "human_visual_decision": None,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "records": review_records,
    }
    technical = {
        "schema_version": "canondressgs.subject00.o03_frozen_contract_canary.technical_report.v1",
        "task_id": TASK_ID,
        "attempt_namespace": ATTEMPT_NAMESPACE,
        "attempt_root": str(ATTEMPT_ROOT),
        "generation_calls": manifest["generation_calls"],
        "output_count": manifest["output_count"],
        "png_parse_pass_count": manifest["png_parse_pass_count"],
        "native_resolution_pass_count": manifest["native_resolution_pass_count"],
        "duplicate_sha_count": manifest["duplicate_sha_count"],
        "old_attempt_comparison": old_attempt_comparison,
        "contact_sheet_path": str(contact_sheet) if contact_sheet.exists() else None,
        "side_by_side_review_paths": side_by_side_paths,
        "final_classification": final_classification,
        "next_task": next_task,
        "paper_modifications": 0,
        "paper_final": False,
    }
    provenance = {
        "schema_version": "canondressgs.subject00.o03_frozen_contract_canary.provenance_registry.v1",
        "task_id": TASK_ID,
        "generation_provider": PROVIDER,
        "generation_mode": MODE,
        "tool_name": TOOL_NAME,
        "external_api_used": False,
        "api_key_used": False,
        "backend_model": None,
        "backend_model_status": BACKEND_MODEL_STATUS,
        "records": [
            {
                "request_id": record["request_id"],
                "source_condition_path": record["source_condition_path"],
                "source_condition_sha256": record["source_condition_sha256"],
                "prompt_path": record["prompt_path"],
                "prompt_sha256": record["prompt_sha256"],
                "output": record.get("output"),
                "status": record["status"],
                "postprocessing_applied": record.get("postprocessing_applied", False),
                "retry_count": record.get("retry_count", 0),
            }
            for record in manifest["records"]
        ],
    }
    summary = {
        "schema_version": "canondressgs.subject00.o03_frozen_contract_canary.final_summary.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "authoritative_canary_contract": "FROZEN_CONTRACT_AT_SOURCE_HEAD",
        "frozen_contract_path": str(FROZEN_CONTRACT_REL),
        "frozen_manifest_path": str(FROZEN_MANIFEST_REL),
        "new_branch": EXECUTION_BRANCH,
        "new_worktree": str(repo),
        "attempt_namespace": ATTEMPT_NAMESPACE,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "generation_authorized": True,
        "request_count": 4,
        "request_ids": EXPECTED_REQUEST_IDS,
        "generation_method": {"generation_provider": PROVIDER, "generation_mode": MODE, "tool_name": TOOL_NAME},
        "generation_calls": manifest["generation_calls"],
        "output_count": manifest["output_count"],
        "output_paths": [record["output"]["path"] for record in successes],
        "output_resolutions": [f"{record['output']['width']}x{record['output']['height']}" for record in successes],
        "output_sha256": [record["output"]["sha256"] for record in successes],
        "technical_report_path": str(repo / TECHNICAL_REPORT_REL),
        "review_manifest_path": str(repo / REVIEW_MANIFEST_REL),
        "contact_sheet_path": str(contact_sheet) if contact_sheet.exists() else None,
        "side_by_side_review_paths": list(side_by_side_paths.values()),
        "human_visual_decision": None,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "paper_modifications": 0,
        "paper_final": False,
        "final_classification": final_classification,
        "next_task": next_task,
    }
    manifest["status"] = final_classification
    manifest["finalized_at"] = now()
    manifest["review_assets"] = {"contact_sheet_path": technical["contact_sheet_path"], "side_by_side_review_paths": side_by_side_paths, "crop_paths": crop_paths}
    atomic_json(manifest_path, manifest)
    atomic_json(ATTEMPT_ROOT / "07_human_review" / "human_review_manifest.json", review_manifest)
    atomic_json(ATTEMPT_ROOT / "08_technical_validation" / "technical_report.json", technical)
    atomic_json(ATTEMPT_ROOT / "09_final" / "final_summary.json", summary)
    atomic_json(repo / EXECUTION_MANIFEST_REL, manifest)
    atomic_json(repo / PROVENANCE_REGISTRY_REL, provenance)
    atomic_json(repo / REVIEW_MANIFEST_REL, review_manifest)
    atomic_json(repo / FINAL_SUMMARY_REL, summary)
    atomic_json(repo / HANDOFF_REL, {"task_id": TASK_ID, "execution_status": final_classification, "attempt_root": str(ATTEMPT_ROOT), "review_manifest": str(repo / REVIEW_MANIFEST_REL), "technical_report": str(repo / TECHNICAL_REPORT_REL), "next_task": next_task})
    report_lines = [
        "# Subject00 O03 Frozen-Contract Canary Technical Report",
        "",
        f"- Task ID: `{TASK_ID}`",
        f"- Source: `{SOURCE_BRANCH}` @ `{SOURCE_HEAD}`",
        f"- Attempt namespace: `{ATTEMPT_NAMESPACE}`",
        f"- Authorization scope: `{AUTHORIZATION_SCOPE}`",
        f"- Generation provider: `{PROVIDER}` / `{MODE}`",
        f"- Generation calls: `{manifest['generation_calls']}`",
        f"- Output count: `{manifest['output_count']}`",
        f"- PNG parse pass: `{manifest['png_parse_pass_count']}`",
        f"- Native resolution pass: `{manifest['native_resolution_pass_count']}`",
        f"- Duplicate SHA count: `{manifest['duplicate_sha_count']}`",
        f"- Attempt 001 mutations: `{old_attempt_comparison['attempt_001']['mutations']}`",
        f"- Attempt 002 mutations: `{old_attempt_comparison['attempt_002']['mutations']}`",
        f"- Attempt 003 mutations: `{old_attempt_comparison['attempt_003']['mutations']}`",
        f"- Contact sheet: `{technical['contact_sheet_path']}`",
        f"- Final classification: `{final_classification}`",
        f"- Next task: `{next_task}`",
        "",
        "Technical validation does not imply human visual acceptance. All human visual fields remain null; accepted and Teacher-target counts remain zero.",
    ]
    atomic_text(repo / TECHNICAL_REPORT_REL, "\n".join(report_lines))
    return summary


def validate_final(repo: Path) -> dict[str, Any]:
    summary = read_json(repo / FINAL_SUMMARY_REL)
    review = read_json(repo / REVIEW_MANIFEST_REL)
    technical_text = (repo / TECHNICAL_REPORT_REL).read_text(encoding="utf-8")
    checks = []
    checks.append(("task_id", summary["task_id"] == TASK_ID))
    checks.append(("authorization_scope", summary["authorization_scope"] == AUTHORIZATION_SCOPE))
    checks.append(("request_count", summary["request_count"] == 4))
    checks.append(("human_null", summary["human_visual_decision"] is None and all(r["human_visual_decision"] is None for r in review["records"])))
    checks.append(("no_promotions", summary["accepted_count"] == 0 and summary["teacher_target_count"] == 0))
    checks.append(("paper_flags", summary["paper_modifications"] == 0 and summary["paper_final"] is False))
    checks.append(("technical_report_mentions_classification", summary["final_classification"] in technical_text))
    failures = [name for name, passed in checks if not passed]
    return {"status": "PASS" if not failures else "FAIL", "checks": [{"name": name, "passed": passed} for name, passed in checks], "failures": failures}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "next", "record-success", "record-failure", "finalize", "validate-final"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--request-id")
    parser.add_argument("--tool-output", type=Path)
    parser.add_argument("--tool-output-hint")
    parser.add_argument("--error-type")
    parser.add_argument("--detail")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    if args.action == "prepare":
        result = prepare(repo)
    elif args.action == "next":
        result = next_pending()
    elif args.action == "record-success":
        if not args.request_id or not args.tool_output:
            raise ValueError("record-success requires --request-id and --tool-output")
        result = record_success(args.request_id, args.tool_output.resolve(), args.tool_output_hint)
    elif args.action == "record-failure":
        if not args.request_id or not args.error_type:
            raise ValueError("record-failure requires --request-id and --error-type")
        result = record_failure(args.request_id, args.error_type, args.tool_output.resolve() if args.tool_output else None, args.detail)
    elif args.action == "finalize":
        result = finalize(repo)
    else:
        result = validate_final(repo)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not isinstance(result, dict) or result.get("status") != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
