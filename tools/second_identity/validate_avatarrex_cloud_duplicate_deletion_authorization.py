from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTH_PATH = (
    REPO_ROOT
    / "paper_protocol"
    / "storage"
    / "avatarrex_cloud_duplicate_deletion_authorization_20260726.json"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    auth = json.loads(AUTH_PATH.read_text(encoding="utf-8"))

    require(
        auth["task_id"]
        == "AAAI27-SUBJECT00-FORMAL-BASE-ARCHIVE-RECLAIM-AND-EXECUTION-001",
        "unexpected task id",
    )
    require(
        auth["source_gate"]["required_source_head"]
        == "e5c98482160b2625a9ac6e8af388bc5350eced46",
        "source head changed",
    )
    require(auth["source_gate"]["training_authorized_before_this_task"] is False, "pre-task training state changed")
    require(auth["source_gate"]["optimizer_steps_before_this_task"] == 0, "pre-task optimizer step count changed")
    require(auth["source_gate"]["output_root_created_before_this_task"] is False, "pre-task output root state changed")

    expected_sha = "531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1"
    expected_bytes = 12569755256
    windows = auth["windows_mirror_verification"]
    cloud = auth["cloud_duplicate_verification"]
    require(windows["primary_mirror_exists"] is True, "primary Windows mirror missing")
    require(windows["primary_mirror_bytes"] == expected_bytes, "primary Windows mirror size changed")
    require(windows["primary_mirror_sha256"] == expected_sha, "primary Windows mirror SHA changed")
    require(cloud["target_path"] == "/root/autodl-tmp/avatarrex_lbn1.7z", "cloud target path changed")
    require(cloud["realpath"] == cloud["target_path"], "cloud realpath changed")
    require(cloud["actual_bytes"] == expected_bytes, "cloud archive size changed")
    require(cloud["actual_sha256"] == expected_sha, "cloud archive SHA changed")
    require(cloud["integrity_result"] == "PASS_EVERYTHING_IS_OK", "7z test did not pass")

    active = auth["active_use_gate"]
    require(active["process_scan_result"] == "PASS_EMPTY", "active process scan not empty")
    require(active["status"] == "PASS_NO_ACTIVE_PROCESS_AND_STAGING_IMMUTABLE", "staging gate did not pass")
    for row in active["staging_paths"]:
        require(row["file_count_before"] == row["file_count_after"], f"file count changed for {row['path']}")
        require(row["logical_bytes_before"] == row["logical_bytes_after"], f"logical bytes changed for {row['path']}")
        require(row["latest_mtime_before"] == row["latest_mtime_after"], f"latest mtime changed for {row['path']}")

    policy = auth["deletion_policy"]
    require(policy["allowed_delete_path"] == "/root/autodl-tmp/avatarrex_lbn1.7z", "allowed delete path changed")
    require(policy["deletion_command_draft"] == "rm -- /root/autodl-tmp/avatarrex_lbn1.7z", "delete command changed")
    forbidden_flags = [
        "delete_windows_mirror",
        "delete_staging",
        "delete_other_cloud_files",
        "extract_avatarrex",
        "full_avatar_o03",
        "gs_vton",
        "teacher_endpoint",
        "paper_modifications",
        "kill_processes",
    ]
    for flag in forbidden_flags:
        require(policy[flag] is False, f"forbidden flag {flag} changed")

    storage = auth["storage_before_deletion"]
    require(storage["projected_peak_write_bytes"] == 14231374378, "projected peak changed")
    require(storage["formal_storage_gate_bytes"] == 32212254720, "formal gate changed")
    require(storage["operational_margin_bytes"] == 2147483648, "operational margin changed")
    require(
        storage["projected_minimum_free_bytes_after_deletion"]
        >= storage["required_projected_minimum_free_bytes_after_deletion"],
        "projected storage after deletion fails gate",
    )

    require(auth["deletion_authorized"] is True, "deletion authorization missing")
    require(auth["deletion_executed"] is False, "pre-delete evidence must not execute deletion")
    require(auth["training_authorized"] is False, "authorization bundle must not authorize training")
    require(auth["optimizer_steps"] == 0, "authorization bundle must not record optimizer steps")
    require(auth["output_root_created"] is False, "authorization bundle must not create output root")

    print("AVATARREX_CLOUD_DUPLICATE_DELETION_AUTHORIZATION_PASS")


if __name__ == "__main__":
    main()
