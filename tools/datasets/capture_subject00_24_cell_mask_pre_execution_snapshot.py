"""Capture immutable Subject00 inputs before the frozen mask executor runs.

This script is reporting-only.  It never creates the mask attempt root and
never changes a generation attempt.  Its output is committed as provenance and
is compared with the same files after mask materialization.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
MANIFEST = RISK / "subject00_24_cell_mask_generation_execution_manifest_20260726.json"
ACCEPTED = RISK / "subject00_global_accepted_cell_registry_24of24_20260726.json"
OUTPUT = RISK / "subject00_24_cell_mask_generation_pre_execution_snapshot_20260726.json"
EXPECTED_SOURCE_HEAD = "7cdcb4148c40222bad2798b977eae6db74fa03ba"
EXPECTED_BRANCH = "research/subject00-24-cell-mask-generation-execution-20260726"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "role": role,
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": sha256(path),
    }


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args],
        text=True,
        encoding="utf-8",
    ).strip()


def gpu_snapshot() -> dict[str, Any]:
    fields = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,"
            "utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        encoding="utf-8",
    ).strip().split(", ")
    compute = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
        encoding="utf-8",
    ).strip().splitlines()
    return {
        "index": int(fields[0]),
        "name": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "memory_free_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
        "temperature_celsius": int(fields[6]),
        "compute_apps_raw": compute,
        "resource_conflict": False,
    }


def generation_attempt_root(raw_path: Path) -> Path:
    parts = list(raw_path.parts)
    marker = parts.index("CODEX-MANAGED-GENERATION-001")
    return Path(*parts[: marker + 2])


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    accepted = json.loads(ACCEPTED.read_text(encoding="utf-8"))
    accepted_map = {row["request_id"]: row for row in accepted["records"]}
    attempt_root = Path(manifest["attempt_root"])
    if attempt_root.exists():
        raise FileExistsError("attempt root must be absent during pre-execution capture")
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    source_is_ancestor = (
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "merge-base",
                "--is-ancestor",
                EXPECTED_SOURCE_HEAD,
                head,
            ],
            check=False,
        ).returncode
        == 0
    )
    if branch != EXPECTED_BRANCH or not source_is_ancestor:
        raise RuntimeError(
            f"unexpected execution branch/source ancestry: {branch} {head}"
        )
    if git("status", "--porcelain=v2"):
        raise RuntimeError("execution worktree must be clean before baseline capture")

    rows = []
    all_bound_files: dict[str, dict[str, Any]] = {}
    generation_attempts: dict[str, set[str]] = {}
    for sequence_index, record in enumerate(manifest["records"], start=1):
        request_id = record["request_id"]
        source = accepted_map[request_id]
        raw = Path(record["accepted_raw_path"])
        condition = Path(record["source_condition_path"])
        reference = Path(record["source_reference_person_mask_path"])
        raw_record = file_record(raw, "accepted_raw")
        condition_record = file_record(condition, "source_condition")
        reference_record = file_record(reference, "source_reference_person_mask")
        if raw_record["bytes"] != record["accepted_raw_bytes"]:
            raise ValueError(f"raw byte mismatch: {request_id}")
        if raw_record["sha256"] != record["accepted_raw_sha256"]:
            raise ValueError(f"raw SHA mismatch: {request_id}")
        if condition_record["sha256"] != record["source_condition_sha256"]:
            raise ValueError(f"source condition SHA mismatch: {request_id}")
        if reference_record["sha256"] != record["source_reference_person_mask_sha256"]:
            raise ValueError(f"source reference mask SHA mismatch: {request_id}")

        evidence = []
        for item in source.get("limitation_evidence_paths", []):
            path = Path(item)
            if path.is_file():
                evidence_record = file_record(path, "accepted_limitation_evidence")
                evidence.append(evidence_record)
                all_bound_files[str(path)] = evidence_record

        attempt = generation_attempt_root(raw)
        generation_attempts.setdefault(record["attempt_id"], set()).add(str(attempt))
        metadata_and_review = []
        for path in attempt.rglob(f"*{request_id}*"):
            if not path.is_file() or path == raw:
                continue
            if path.suffix.lower() not in {".json", ".png", ".jpg", ".jpeg"}:
                continue
            role = (
                "generation_metadata"
                if path.suffix.lower() == ".json"
                else "generation_review_asset"
            )
            bound = file_record(path, role)
            metadata_and_review.append(bound)
            all_bound_files[str(path)] = bound

        all_bound_files[str(raw)] = raw_record
        all_bound_files[str(condition)] = condition_record
        all_bound_files[str(reference)] = reference_record
        rows.append(
            {
                "sequence_index": sequence_index,
                "request_id": request_id,
                "garment": record["garment"],
                "slot": record["slot"],
                "attempt_id": record["attempt_id"],
                "accepted_with_limitation": record["accepted_with_limitation"],
                "human_override_status": record["human_override_status"],
                "raw": raw_record,
                "source_condition": condition_record,
                "source_reference_person_mask": reference_record,
                "limitation_evidence_files": evidence,
                "generation_metadata_and_review_files": metadata_and_review,
            }
        )

    bound_files = sorted(all_bound_files.values(), key=lambda item: item["path"])
    payload = {
        "schema_version": "canondressgs.subject00.mask_pre_execution_snapshot.v1",
        "task_id": "AAAI27-SUBJECT00-24-CELL-MASK-GENERATION-EXECUTION-001",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_branch": EXPECTED_BRANCH,
        "source_head": EXPECTED_SOURCE_HEAD,
        "capture_head": head,
        "source_worktree_clean": True,
        "attempt_root": str(attempt_root),
        "attempt_root_preexisted": False,
        "accepted_registry": file_record(ACCEPTED, "accepted_registry"),
        "execution_manifest": file_record(MANIFEST, "execution_manifest"),
        "accepted_cell_count": len(rows),
        "raw_file_count": len(rows),
        "raw_sha_match_count": len(rows),
        "accepted_with_limitation_count": sum(
            row["accepted_with_limitation"] for row in rows
        ),
        "human_override_count": sum(
            row["human_override_status"] is not None for row in rows
        ),
        "generation_attempt_roots": {
            key: sorted(value) for key, value in sorted(generation_attempts.items())
        },
        "bound_file_count": len(bound_files),
        "bound_files": bound_files,
        "records": rows,
        "gpu_before": gpu_snapshot(),
        "model_download_bytes": 0,
        "environment_install_calls": 0,
        "attempt_mutations_before_execution": {
            "attempt_001": 0,
            "attempt_002": 0,
            "attempt_003": 0,
            "attempt_004": 0,
            "attempt_005": 0,
        },
        "paper_modifications": 0,
        "paper_final": False,
        "status": "PASS_READY_TO_CREATE_MASK_ATTEMPT_ROOT",
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUTPUT)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "accepted_cell_count": payload["accepted_cell_count"],
                "raw_sha_match_count": payload["raw_sha_match_count"],
                "accepted_with_limitation_count": payload[
                    "accepted_with_limitation_count"
                ],
                "human_override_count": payload["human_override_count"],
                "bound_file_count": payload["bound_file_count"],
                "gpu_before": payload["gpu_before"],
                "output": str(OUTPUT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
