"""State manager for the six-call Subject00 V2 execution.

This script never calls an image model. It initializes the frozen attempt,
records one already-returned native PNG at a time, and writes provenance
immediately. It fails closed on call order, retries, paths, and source hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
MANIFEST_PATH = RISK / "subject00_remaining_six_generation_execution_manifest_v2_20260726.json"
PROJECT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001")
ATTEMPT = PROJECT / "attempt_005_subject00_remaining_six_cell_generation"
STATE = ATTEMPT / "07_logs" / "execution_state.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def inventory(path: Path) -> dict:
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.exists() else []
    key_names = {
        "execution_manifest.json", "continuation_state.json", "generation_record.json",
        "subject00_o03_canary_human_review_overlay_20260726.json",
    }
    return {
        "path": str(path), "exists": path.exists(), "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "key_file_sha256": {str(p): sha(p) for p in files if p.name in key_names},
    }


def init() -> None:
    if ATTEMPT.exists():
        raise RuntimeError("attempt root already exists")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["generation_calls"] != 0 or manifest["authorization_remaining_calls"] != 6:
        raise RuntimeError("authorization is not pristine")
    for record in manifest["records"]:
        source = Path(record["windows_source_condition_path"])
        if sha(source) != record["source_condition_sha256"]:
            raise RuntimeError(f"source SHA mismatch: {record['request_id']}")
        with Image.open(source) as image:
            image.verify()
    for value in manifest["directory_layout"].values():
        Path(value).mkdir(parents=True, exist_ok=False)
    snapshot = Path(manifest["directory_layout"]["contract_snapshot_root"])
    for source in [
        RISK / "SUBJECT00_REMAINING_SIX_GENERATION_EXECUTION_CONTRACT_V2_20260726.md",
        MANIFEST_PATH,
        RISK / "subject00_remaining_six_execution_contract_v2_correction_overlay_20260726.json",
    ]:
        shutil.copy2(source, snapshot / source.name)
    write_json(snapshot / "authorization_registry.json", {
        "authorization_source": manifest["authorization_source"],
        "authorization_scope": manifest["authorization_scope"],
        "calls_before": 6, "calls_consumed": 0, "calls_remaining": 6,
        "status": "VALID_UNCONSUMED", "generation_executed": False,
    })
    source_bindings = [{
        "request_id": r["request_id"], "path": r["windows_source_condition_path"],
        "sha256": r["source_condition_sha256"], "camera": r["camera"], "frame": "00000000",
    } for r in manifest["records"]]
    write_json(Path(manifest["directory_layout"]["source_binding_root"]) / "source_binding_registry.json", source_bindings)
    old = {name: inventory(PROJECT / name) for name in [
        "attempt_001", "attempt_002", "attempt_003", "attempt_004_o03_hood_removal_targeted_canary"
    ]}
    write_json(Path(manifest["directory_layout"]["audit_root"]) / "old_attempts_before.json", old)
    write_json(STATE, {
        "schema_version": "canondressgs.subject00.remaining_six.execution_state.v1",
        "task_id": "AAAI27-SUBJECT00-REMAINING-SIX-V2-CONTRACT-GENERATION-EXECUTION-001",
        "started_at": datetime.now(timezone.utc).isoformat(), "generation_calls": 0,
        "retry_calls": 0, "postprocessing_calls": 0, "authorization_calls_before": 6,
        "authorization_calls_consumed": 0, "authorization_calls_remaining": 6,
        "records": [], "status": "INITIALIZED", "paper_final": False,
    })
    requests = Path(manifest["directory_layout"]["request_root"])
    for r in manifest["records"]:
        write_json(requests / f"{r['request_id']}.json", r)


def record(args: argparse.Namespace) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(STATE.read_text(encoding="utf-8"))
    index = state["generation_calls"]
    if index >= 6 or args.request_id != manifest["execution_order"][index]:
        raise RuntimeError("request order/call budget violation")
    request = manifest["records"][index]
    returned = Path(args.returned_path)
    with Image.open(returned) as image:
        image.verify()
        width, height = image.size
        fmt = image.format
    allowed = {(x["width"], x["height"]) for x in manifest["accepted_resolution_set"]}
    valid = fmt == "PNG" and (width, height) in allowed
    destination = Path(request["expected_raw_output_path"] if valid else request["technical_failure_path"])
    if destination.exists():
        raise RuntimeError("destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(returned, destination)
    output_sha = sha(destination)
    duplicate = output_sha in {r["output_sha256"] for r in state["records"]}
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "request_id": args.request_id, "call_index": index + 1,
        "start_time": args.start_time, "end_time": now,
        "source_path": request["windows_source_condition_path"],
        "source_sha256": request["source_condition_sha256"],
        "output_path": str(destination), "output_bytes": destination.stat().st_size,
        "output_sha256": output_sha, "raw_resolution": {"width": width, "height": height},
        "png_parse": fmt == "PNG", "resolution_family_pass": valid,
        "duplicate_sha": duplicate, "generation_response": args.response,
        "no_postprocessing": True, "retry_calls": 0, "postprocessing_calls": 0,
        "status": "RAW_VALID_PENDING_REGISTRATION" if valid else "TECHNICAL_FAILURE_RESOLUTION",
    }
    write_json(Path(request["response_metadata_path"]), payload)
    state["records"].append(payload)
    state["generation_calls"] += 1
    state["authorization_calls_consumed"] += 1
    state["authorization_calls_remaining"] -= 1
    state["status"] = "PARTIAL_STOP_RESOLUTION_FAILURE" if not valid else "IN_PROGRESS"
    write_json(STATE, state)
    print(json.dumps(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    rec = sub.add_parser("record")
    rec.add_argument("--request-id", required=True)
    rec.add_argument("--returned-path", required=True)
    rec.add_argument("--start-time", required=True)
    rec.add_argument("--response", required=True)
    args = parser.parse_args()
    init() if args.command == "init" else record(args)


if __name__ == "__main__":
    main()
