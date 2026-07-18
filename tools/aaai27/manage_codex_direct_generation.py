#!/usr/bin/env python3
"""Manage append-only Codex-native target generation evidence.

This tool never generates an image and has no network or credential surface.
Codex invokes the platform-managed ``image_gen.imagegen`` tool directly; this
manager freezes inputs/prompts, copies the first technically valid tool output
byte-for-byte, records provenance, resumes at the first incomplete sample, and
builds/verifies the minimal handoff after all 28 samples pass visual review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


TASK_ID = "SUBJECT02-AAAI27-CODEX-DIRECT-GENERATION-GATE-002"
RECORD_SCHEMA = "canondressgs.aaai27.codex_generation_record.v1"
PROVIDER = "CODEX_IMAGE_GENERATION_SKILL"
MODE = "CODEX_PLATFORM_MANAGED_DIRECT_IMAGE_EDIT"
CLIENT_SURFACE = "VS_CODE_IDE_EXTENSION"
SKILL_NAME = "imagegen"
TOOL_NAME = "image_gen.imagegen"
BACKEND_MODEL_STATUS = "NOT_EXPOSED_BY_PLATFORM"
EXPECTED_LONG_HEAD = "9fc88033b407136073f7bddc6ffca6dd4dd1e0f5"
EXPECTED_BRANCH = "sprint/aaai27-20260718"
EXPECTED_OUTFITS = ("O01", "O02", "O03", "O04", "O06", "O07", "O08")
EXPECTED_CONDITIONS = (
    ("cond_000000", "front"),
    ("cond_000318", "back"),
    ("cond_000017", "left"),
    ("cond_000347", "right"),
)
INPUT_ROLES = (
    "subject02_base_render",
    "subject02_clay_condition",
    "Jay_outfit_reference",
)
PRIMARY_INVOCATION_MAX = 28
TECHNICAL_RETRY_MAX = 4
TOTAL_INVOCATION_MAX = 32
EXPECTED_SIZE = (1024, 1536)
DEFAULT_PLAN = Path(
    r"E:\data_pre\audit_aaai27_data_capacity_gate_001\attempt_001\generation\generation_plan.json"
)
FORBIDDEN_SECRET_MARKERS = (
    b"Authorization: Bearer ",
    b'"authorization"',
    b"OPENAI_API_KEY=",
    b"SUBLYX_API_KEY=",
    b"GPT_IMAGE_API_KEY=",
    b"IMAGE_API_KEY=",
)


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text_exact(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8").strip()


def assert_git_contract(repo: Path, require_clean: bool = True) -> dict[str, Any]:
    state = {
        "branch": git(repo, "branch", "--show-current"),
        "head": git(repo, "rev-parse", "HEAD"),
        "long_term_head": git(repo, "rev-parse", "refs/heads/pipeline/full-dressable-20260715"),
        "status_short": git(repo, "status", "--short"),
    }
    if state["branch"] != EXPECTED_BRANCH:
        raise RuntimeError(f"Wrong branch: {state['branch']}")
    if state["long_term_head"] != EXPECTED_LONG_HEAD:
        raise RuntimeError(f"Long-term branch changed: {state['long_term_head']}")
    if require_clean and state["status_short"]:
        raise RuntimeError(f"Worktree must be clean:\n{state['status_short']}")
    return state


def validate_generation_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    records = plan.get("records", [])
    expected = {(outfit, condition) for outfit in EXPECTED_OUTFITS for condition, _ in EXPECTED_CONDITIONS}
    actual = {(record["outfit_id"], record["condition_id"]) for record in records}
    if len(records) != 28 or len(actual) != 28 or actual != expected:
        raise ValueError("Generation plan must contain the exact 7x4 preregistered samples")
    for record in records:
        if record.get("target_used_as_forward_condition") is not False:
            raise ValueError(f"Target leakage in {record['sample_id']}")
        if len(record.get("inputs", {})) != 5:
            raise ValueError(f"Incomplete frozen inputs for {record['sample_id']}")
        if sha256_text(record["prompt"]) != record["prompt_sha256"]:
            raise ValueError(f"Prompt hash mismatch for {record['sample_id']}")
    return records


def existing_target_is_reusable(record: dict[str, Any]) -> bool:
    """Only a result already governed by this exact provider contract is reusable."""

    return bool(
        record.get("generation_provider") == PROVIDER
        and record.get("external_api_used") is False
        and record.get("api_key_used") is False
        and record.get("raw_output_immutable") is True
        and record.get("output_sha256")
    )


def backend_model_contract_valid(model: str | None, status: str) -> bool:
    return bool(model) or (model is None and status == BACKEND_MODEL_STATUS)


def handoff_counts_are_complete(raw_target_names: Iterable[str], record_names: Iterable[str]) -> bool:
    targets = list(raw_target_names)
    records = list(record_names)
    return len(targets) == len(set(targets)) == 28 and len(records) == len(set(records)) == 28


def input_entries(source: dict[str, Any]) -> list[dict[str, Any]]:
    role_map = (
        ("subject02_base_render", source["inputs"]["base"]),
        ("subject02_clay_condition", source["inputs"]["clay"]),
        ("Jay_outfit_reference", source["inputs"]["reference"]),
    )
    entries = []
    for role, item in role_map:
        path = Path(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen input mismatch for {source['sample_id']} / {role}: {path}")
        entries.append({"role": role, "path": str(path.resolve()), "sha256": item["sha256"]})
    return entries


def generation_record_path(attempt: Path, sample: dict[str, Any]) -> Path:
    return attempt / "provenance" / sample["sample_id"] / "generation_record.json"


def raw_target_path(attempt: Path, sample: dict[str, Any]) -> Path:
    return attempt / "raw_targets" / sample["outfit_id"] / sample["condition_id"] / "raw_direct_edit.png"


def prepare_attempt(repo: Path, attempt: Path, plan_path: Path) -> dict[str, Any]:
    if attempt.exists():
        raise FileExistsError(f"Append-only attempt already exists: {attempt}")
    git_state = assert_git_contract(repo)
    source_plan = read_json(plan_path)
    source_records = validate_generation_plan(source_plan)
    for name in (
        "contract", "inputs", "prompts", "raw_targets", "provenance", "visual_gate",
        "contact_sheets", "handoff", "final",
    ):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    records = []
    input_manifest = []
    prompt_hashes = []
    for index, source in enumerate(source_records):
        inputs = input_entries(source)
        prompt_path = attempt / "prompts" / f"{source['sample_id']}.txt"
        atomic_text_exact(prompt_path, source["prompt"])
        prompt_hash = sha256(prompt_path)
        if prompt_hash != source["prompt_sha256"]:
            raise RuntimeError(f"Written prompt hash mismatch: {source['sample_id']}")
        sample = {
            "index": index,
            "sample_id": source["sample_id"],
            "outfit_id": source["outfit_id"],
            "condition_id": source["condition_id"],
            "view": source["view"],
            "status": "PENDING",
            "generation_provider": PROVIDER,
            "inputs": inputs,
            "prompt_path": str(prompt_path.resolve()),
            "prompt_sha256": prompt_hash,
            "prompt_source": source["prompt_source"],
            "raw_target_path": str(raw_target_path(attempt, source).resolve()),
            "generation_record_path": str(generation_record_path(attempt, source).resolve()),
            "invocation_count": 0,
            "retry_count": 0,
        }
        records.append(sample)
        input_manifest.append({"sample_id": source["sample_id"], "inputs": inputs})
        prompt_hashes.append({"sample_id": source["sample_id"], "path": str(prompt_path.resolve()), "sha256": prompt_hash})
    contract = {
        "schema_version": "canondressgs.aaai27.codex_direct_generation.attempt.v1",
        "task_id": TASK_ID,
        "created_at": now(),
        "git": git_state,
        "generation_provider": PROVIDER,
        "generation_mode": MODE,
        "client": "Codex",
        "client_surface": CLIENT_SURFACE,
        "skill_name": SKILL_NAME,
        "tool_name": TOOL_NAME,
        "platform_managed": True,
        "external_api_used": False,
        "api_key_used": False,
        "backend_model": None,
        "backend_model_status": BACKEND_MODEL_STATUS,
        "candidate_sampling": False,
        "existing_unknown_model_targets_reused": 0,
        "input_roles": list(INPUT_ROLES),
        "primary_invocation_max": PRIMARY_INVOCATION_MAX,
        "technical_retry_max": TECHNICAL_RETRY_MAX,
        "total_invocation_max": TOTAL_INVOCATION_MAX,
        "expected_size": list(EXPECTED_SIZE),
        "source_generation_plan": {"path": str(plan_path.resolve()), "sha256": sha256(plan_path)},
    }
    manifest = {
        "schema_version": "canondressgs.aaai27.codex_direct_generation.manifest.v1",
        "task_id": TASK_ID,
        "status": "READY",
        "record_count": 28,
        "successful_count": 0,
        "primary_invocations": 0,
        "technical_retries": 0,
        "total_invocations": 0,
        "records": records,
    }
    atomic_json(attempt / "contract" / "generation_contract.json", contract)
    atomic_json(attempt / "inputs" / "input_sha256_manifest.json", {"records": input_manifest})
    atomic_json(attempt / "prompts" / "prompt_hashes.json", {"records": prompt_hashes})
    atomic_json(attempt / "provenance" / "generation_manifest.json", manifest)
    append_jsonl(attempt / "provenance" / "state_records.jsonl", {"timestamp": now(), "event": "ATTEMPT_PREPARED", "successful_count": 0})
    atomic_json(attempt / "final" / "start_state.json", {"status": "READY", "git": git_state, "network_requests": 0, "optimizer_steps": 0})
    return {"status": "READY", "record_count": 28, "attempt": str(attempt)}


def load_manifest(attempt: Path) -> tuple[Path, dict[str, Any]]:
    path = attempt / "provenance" / "generation_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing generation manifest: {path}")
    return path, read_json(path)


def next_pending_record(manifest: dict[str, Any]) -> dict[str, Any] | None:
    for record in manifest["records"]:
        if record["status"] in {"PENDING", "TECHNICAL_RETRY_ALLOWED"}:
            return record
    return None


def check_budget(manifest: dict[str, Any], is_retry: bool) -> None:
    primary = int(manifest["primary_invocations"])
    retries = int(manifest["technical_retries"])
    total = int(manifest["total_invocations"])
    if is_retry and retries >= TECHNICAL_RETRY_MAX:
        raise RuntimeError("Technical retry budget exhausted")
    if not is_retry and primary >= PRIMARY_INVOCATION_MAX:
        raise RuntimeError("Primary invocation budget exhausted")
    if total >= TOTAL_INVOCATION_MAX:
        raise RuntimeError("Total invocation budget exhausted")


def image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        if image.size != EXPECTED_SIZE:
            raise ValueError(f"Expected {EXPECTED_SIZE}, got {image.size}")
        if image.format != "PNG":
            raise ValueError(f"Expected native PNG output, got {image.format}")
        extrema = image.convert("RGB").getextrema()
        if all(low == high for low, high in extrema):
            raise ValueError("Output is a blank/constant image")
        return {"width": image.width, "height": image.height, "mode": image.mode, "format": image.format}


def copy_immutable_output(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Raw output is immutable: {destination}")
    metadata = image_metadata(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != sha256(source):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Byte-copy hash mismatch")
    temporary.replace(destination)
    metadata.update({"size_bytes": destination.stat().st_size, "sha256": sha256(destination)})
    return metadata


def record_success(attempt: Path, sample_id: str, source_image: Path, tool_output_hint: str | None) -> dict[str, Any]:
    manifest_path, manifest = load_manifest(attempt)
    matches = [record for record in manifest["records"] if record["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate sample: {sample_id}")
    sample = matches[0]
    if sample["status"] not in {"PENDING", "TECHNICAL_RETRY_ALLOWED"}:
        raise RuntimeError(f"Sample is not pending: {sample_id} / {sample['status']}")
    is_retry = sample["status"] == "TECHNICAL_RETRY_ALLOWED"
    check_budget(manifest, is_retry)
    output_path = Path(sample["raw_target_path"])
    output = copy_immutable_output(source_image, output_path)
    generation_record = {
        "schema_version": RECORD_SCHEMA,
        "task_id": TASK_ID,
        "sample_id": sample_id,
        "outfit_id": sample["outfit_id"],
        "condition_id": sample["condition_id"],
        "view": sample["view"],
        "generation_provider": PROVIDER,
        "generation_mode": MODE,
        "client": "Codex",
        "client_surface": CLIENT_SURFACE,
        "skill_name": SKILL_NAME,
        "tool_name": TOOL_NAME,
        "platform_managed": True,
        "external_api_used": False,
        "api_key_used": False,
        "backend_model": None,
        "backend_model_status": BACKEND_MODEL_STATUS,
        "inputs": sample["inputs"],
        "prompt_path": sample["prompt_path"],
        "prompt_sha256": sample["prompt_sha256"],
        "generation_timestamp": now(),
        "codex_task_session_identifier": None,
        "codex_task_session_identifier_status": "NOT_EXPOSED_TO_TOOL",
        "tool_output_hint": tool_output_hint,
        "output": {"path": str(output_path.resolve()), **output},
        "retry_count": sample["retry_count"],
        "raw_output_immutable": True,
    }
    record_path = Path(sample["generation_record_path"])
    if record_path.exists():
        raise FileExistsError(f"Generation record is append-only: {record_path}")
    atomic_json(record_path, generation_record)
    sample["status"] = "SUCCESS"
    sample["invocation_count"] += 1
    manifest["successful_count"] += 1
    manifest["total_invocations"] += 1
    if is_retry:
        manifest["technical_retries"] += 1
    else:
        manifest["primary_invocations"] += 1
    manifest["status"] = "COMPLETE_AWAITING_VISUAL_GATE" if manifest["successful_count"] == 28 else "PARTIAL_GENERATION_IN_PROGRESS"
    atomic_json(manifest_path, manifest)
    append_jsonl(
        attempt / "provenance" / "state_records.jsonl",
        {"timestamp": now(), "event": "GENERATION_SUCCESS", "sample_id": sample_id, "output_sha256": output["sha256"], "successful_count": manifest["successful_count"]},
    )
    return {"status": sample["status"], "sample_id": sample_id, "output": output, "successful_count": manifest["successful_count"]}


def record_technical_failure(attempt: Path, sample_id: str, error_type: str) -> dict[str, Any]:
    allowed = {"EMPTY_RESULT", "SAVE_FAILURE", "PLATFORM_INTERRUPTED", "DECODE_FAILURE", "TEMPORARY_TOOL_ERROR"}
    if error_type not in allowed:
        raise ValueError(f"Non-technical failure cannot consume retry budget: {error_type}")
    manifest_path, manifest = load_manifest(attempt)
    sample = next(record for record in manifest["records"] if record["sample_id"] == sample_id)
    if sample["status"] not in {"PENDING", "TECHNICAL_RETRY_ALLOWED"}:
        raise RuntimeError(f"Cannot record failure for {sample['status']}")
    is_retry = sample["status"] == "TECHNICAL_RETRY_ALLOWED"
    check_budget(manifest, is_retry)
    sample["invocation_count"] += 1
    sample["retry_count"] += 1
    manifest["total_invocations"] += 1
    if is_retry:
        manifest["technical_retries"] += 1
    else:
        manifest["primary_invocations"] += 1
    sample["status"] = "TECHNICAL_RETRY_ALLOWED" if sample["retry_count"] <= 1 and manifest["technical_retries"] < TECHNICAL_RETRY_MAX else "TECHNICAL_FAILURE"
    manifest["status"] = "PARTIAL_GENERATION_PAUSED"
    append_jsonl(attempt / "provenance" / "state_records.jsonl", {"timestamp": now(), "event": "GENERATION_TECHNICAL_FAILURE", "sample_id": sample_id, "error_type": error_type, "retry_count": sample["retry_count"]})
    atomic_json(manifest_path, manifest)
    return {"status": sample["status"], "sample_id": sample_id, "retry_count": sample["retry_count"]}


def validate_complete_records(attempt: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if len(manifest["records"]) != 28 or manifest["successful_count"] != 28:
        raise RuntimeError("Gate cannot start before 28 valid targets")
    seen = set()
    generation_records = []
    for sample in manifest["records"]:
        if sample["sample_id"] in seen or sample["status"] != "SUCCESS":
            raise RuntimeError("Duplicate or incomplete sample")
        seen.add(sample["sample_id"])
        path = Path(sample["generation_record_path"])
        record = read_json(path)
        output = Path(record["output"]["path"])
        if sha256(output) != record["output"]["sha256"] or image_metadata(output)["width"] != EXPECTED_SIZE[0]:
            raise RuntimeError(f"Immutable output verification failed: {sample['sample_id']}")
        if record["generation_provider"] != PROVIDER or record["external_api_used"] or record["api_key_used"]:
            raise RuntimeError(f"Provenance mismatch: {sample['sample_id']}")
        generation_records.append(record)
    return generation_records


def validate_visual_gate(rows: list[dict[str, Any]], sample_ids: set[str]) -> None:
    if len(rows) != 28 or {row["sample_id"] for row in rows} != sample_ids:
        raise ValueError("Visual gate must contain exactly the 28 generated samples")
    allowed = {"PASS", "WARN", "FAIL"}
    if any(row["status"] not in allowed for row in rows):
        raise ValueError("Unknown visual-gate status")
    if any(not row.get("images_actually_opened") or not row.get("observation") for row in rows):
        raise ValueError("Every image must be actually opened and described")


def build_contact_sheet(records: list[dict[str, Any]], path: Path) -> None:
    cell_width, cell_height, label_height = 256, 384, 34
    canvas = Image.new("RGB", (cell_width * 4, (cell_height + label_height) * 7), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, record in enumerate(records):
        with Image.open(record["output"]["path"]) as image:
            tile = image.convert("RGB").resize((cell_width, cell_height), Image.Resampling.LANCZOS)
        x, y = (index % 4) * cell_width, (index // 4) * (cell_height + label_height)
        canvas.paste(tile, (x, y))
        draw.text((x + 5, y + cell_height + 6), f"{record['outfit_id']} {record['condition_id']} {record['view']}", fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def finalize(attempt: Path, visual_gate_path: Path) -> dict[str, Any]:
    manifest_path, manifest = load_manifest(attempt)
    records = validate_complete_records(attempt, manifest)
    visual_rows = read_json(visual_gate_path)["records"]
    validate_visual_gate(visual_rows, {record["sample_id"] for record in records})
    gate_by_id = {row["sample_id"]: row for row in visual_rows}
    csv_path = attempt / "visual_gate" / "generation_sample_gate.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "outfit_id", "condition_id", "view", "status", "observation"])
        writer.writeheader()
        for record in records:
            row = gate_by_id[record["sample_id"]]
            writer.writerow({key: row[key] for key in writer.fieldnames})
    build_contact_sheet(records, attempt / "contact_sheets" / "generation_28_contact_sheet.png")
    counts = {status: sum(row["status"] == status for row in visual_rows) for status in ("PASS", "WARN", "FAIL")}
    summary = {
        "schema_version": "canondressgs.aaai27.codex_direct_generation.summary.v1",
        "task_id": TASK_ID,
        "status": "PASS" if counts["FAIL"] == 0 else "FAIL",
        "record_count": 28,
        "visual_counts": counts,
        "generation_provider": PROVIDER,
        "provider_consistent": True,
        "external_api_used": False,
        "api_key_used": False,
        "backend_model": None,
        "backend_model_status": BACKEND_MODEL_STATUS,
        "primary_invocations": manifest["primary_invocations"],
        "technical_retries": manifest["technical_retries"],
        "total_invocations": manifest["total_invocations"],
        "raw_outputs_immutable": True,
        "records": [{"sample_id": record["sample_id"], "output_sha256": record["output"]["sha256"], "visual_status": gate_by_id[record["sample_id"]]["status"]} for record in records],
    }
    atomic_json(attempt / "provenance" / "generation_28_provenance_summary.json", summary)
    lines = ["# Generation Visual Gate", "", f"- Status: **{summary['status']}**", f"- PASS/WARN/FAIL: `{counts['PASS']}/{counts['WARN']}/{counts['FAIL']}`", "- All 28 images were actually opened; per-image observations are in `visual_gate_records.json`."]
    atomic_text(attempt / "visual_gate" / "GENERATION_VISUAL_GATE.md", "\n".join(lines))
    shutil.copyfile(visual_gate_path, attempt / "visual_gate" / "visual_gate_records.json")
    manifest["status"] = "GENERATION_GATE_PASS" if summary["status"] == "PASS" else "GENERATION_GATE_FAIL"
    atomic_json(manifest_path, manifest)
    atomic_json(attempt / "final" / "GENERATION_FINAL_STATUS.json", summary)
    return summary


def contains_secret(data: bytes) -> bool:
    lowered = data.lower()
    if any(marker.lower() in lowered for marker in FORBIDDEN_SECRET_MARKERS):
        return True
    return re.search(rb"(?i)(?:bearer\s+|api[_-]?key\s*[:=]\s*)[A-Za-z0-9_\-]{20,}", data) is not None


def allowed_bundle_files(attempt: Path) -> list[Path]:
    files = []
    files.extend(sorted((attempt / "raw_targets").rglob("raw_direct_edit.png")))
    files.extend(sorted((attempt / "provenance").rglob("generation_record.json")))
    files.extend(
        [
            attempt / "prompts" / "prompt_hashes.json",
            attempt / "inputs" / "input_sha256_manifest.json",
            attempt / "visual_gate" / "generation_sample_gate.csv",
            attempt / "visual_gate" / "GENERATION_VISUAL_GATE.md",
            attempt / "visual_gate" / "visual_gate_records.json",
            attempt / "provenance" / "generation_28_provenance_summary.json",
        ]
    )
    return files


def build_bundle(attempt: Path) -> dict[str, Any]:
    _, manifest = load_manifest(attempt)
    validate_complete_records(attempt, manifest)
    summary = read_json(attempt / "provenance" / "generation_28_provenance_summary.json")
    if summary["status"] != "PASS":
        raise RuntimeError("Cannot hand off a failed generation gate")
    files = allowed_bundle_files(attempt)
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("Handoff input missing")
    members = []
    for path in files:
        data = path.read_bytes()
        if contains_secret(data):
            raise RuntimeError(f"Secret marker in handoff file: {path}")
        members.append({"path": str(path.relative_to(attempt)).replace("\\", "/"), "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    bundle_manifest = {
        "schema_version": "canondressgs.aaai27.codex_direct_generation.bundle.v1",
        "task_id": TASK_ID,
        "created_at": now(),
        "generation_provider": PROVIDER,
        "external_api_used": False,
        "api_key_used": False,
        "raw_target_count": 28,
        "generation_record_count": 28,
        "members": members,
    }
    manifest_path = attempt / "handoff" / "bundle_manifest.json"
    atomic_json(manifest_path, bundle_manifest)
    bundle_path = attempt / "handoff" / "codex_direct_generation_gate_28_bundle.tar"
    if bundle_path.exists():
        raise FileExistsError(f"Bundle is append-only: {bundle_path}")
    with tarfile.open(bundle_path, "w") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(attempt)).replace("\\", "/"), recursive=False)
        archive.add(manifest_path, arcname="handoff/bundle_manifest.json", recursive=False)
    result = {"bundle_path": str(bundle_path.resolve()), "bundle_sha256": sha256(bundle_path), "bundle_size": bundle_path.stat().st_size, "member_count": len(members) + 1}
    atomic_json(attempt / "handoff" / "bundle_output_manifest.json", result)
    return result


def verify_bundle(bundle: Path) -> dict[str, Any]:
    with tarfile.open(bundle, "r") as archive:
        names = archive.getnames()
        if len(names) != len(set(names)):
            raise RuntimeError("Duplicate bundle member")
        manifest = json.loads(archive.extractfile("handoff/bundle_manifest.json").read().decode("utf-8"))
        expected = {item["path"]: item for item in manifest["members"]}
        failures = []
        png_count = 0
        record_count = 0
        for name, item in expected.items():
            data = archive.extractfile(name).read()
            if hashlib.sha256(data).hexdigest() != item["sha256"] or len(data) != item["size"] or contains_secret(data):
                failures.append(name)
            if name.endswith("raw_direct_edit.png"):
                png_count += 1
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    if image.size != EXPECTED_SIZE:
                        failures.append(name + ":shape")
            if name.endswith("generation_record.json"):
                record_count += 1
                record = json.loads(data.decode("utf-8"))
                if record["generation_provider"] != PROVIDER or record["external_api_used"] or record["api_key_used"]:
                    failures.append(name + ":provenance")
        return {
            "status": "PASS" if not failures and png_count == 28 and record_count == 28 else "FAIL",
            "bundle_sha256": sha256(bundle),
            "member_count": len(names),
            "raw_target_count": png_count,
            "generation_record_count": record_count,
            "failures": failures,
            "secret_scan": "PASS" if not failures else "CHECK_FAILURES",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "next", "record-success", "record-failure", "finalize", "bundle", "verify-bundle"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--attempt-dir", type=Path)
    parser.add_argument("--generation-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--sample-id")
    parser.add_argument("--source-image", type=Path)
    parser.add_argument("--tool-output-hint")
    parser.add_argument("--error-type")
    parser.add_argument("--visual-gate-json", type=Path)
    parser.add_argument("--bundle", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action != "verify-bundle" and args.attempt_dir is None:
        raise ValueError("--attempt-dir is required")
    if args.action == "prepare":
        result = prepare_attempt(args.repo_root.resolve(), args.attempt_dir.resolve(), args.generation_plan.resolve())
    elif args.action == "next":
        _, manifest = load_manifest(args.attempt_dir.resolve())
        result = next_pending_record(manifest) or {"status": "NO_PENDING_SAMPLES"}
    elif args.action == "record-success":
        if not args.sample_id or not args.source_image:
            raise ValueError("record-success requires --sample-id and --source-image")
        result = record_success(args.attempt_dir.resolve(), args.sample_id, args.source_image.resolve(), args.tool_output_hint)
    elif args.action == "record-failure":
        if not args.sample_id or not args.error_type:
            raise ValueError("record-failure requires --sample-id and --error-type")
        result = record_technical_failure(args.attempt_dir.resolve(), args.sample_id, args.error_type)
    elif args.action == "finalize":
        if not args.visual_gate_json:
            raise ValueError("finalize requires --visual-gate-json")
        result = finalize(args.attempt_dir.resolve(), args.visual_gate_json.resolve())
    elif args.action == "bundle":
        result = build_bundle(args.attempt_dir.resolve())
    else:
        if not args.bundle:
            raise ValueError("verify-bundle requires --bundle")
        result = verify_bundle(args.bundle.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not isinstance(result, dict) or result.get("status") not in {"FAIL", "TECHNICAL_FAILURE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
