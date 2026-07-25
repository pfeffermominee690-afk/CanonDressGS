#!/usr/bin/env python3
"""Run the non-execution gates for Subject00 generation preparation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
SOURCE_HEAD = "9da04d923c7c6c03c6a6f7c732d8d9c58d59c88d"
FORMAL_HEAD = "4993f5c865ec19895f35811fa399fc4a1834c6a7"
STORAGE_HEAD = "34e91445ef45c001cebcd789a4a6a88cad7b9ad8"
TRAIN_CAMERAS = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23}
HELDOUT_CAMERAS = {0, 4, 8, 12, 16, 20}
CLASSIFICATION = "SUBJECT00_DATA_PREPARATION_READY_PENDING_BACKEND"
NEXT_TASK = "USER_ADJUDICATE_SUBJECT00_GENERATION_BACKEND"
DATASET_ROOT = "/root/autodl-tmp/canondressgs_work/datasets/subject00_three_garment"
FORMAL_ROOT = "/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-FORMAL-STRICT-SPLIT-001"
JSON_OUTPUTS = [
    "subject00_data_preparation_recovery_audit.json",
    "subject00_identity_source_registry.json",
    "subject00_condition_asset_inventory.json",
    "subject00_condition_slot_binding.json",
    "subject00_controller_split_blueprint.json",
    "subject00_garment_occlusion_allowlist.json",
    "subject00_garment_reject_taxonomy.json",
    "subject00_generation_backend_selection.json",
    "subject00_generation_prompt_registry.json",
    "subject00_generation_request_manifest.json",
    "subject00_human_review_execution_manifest.json",
    "subject00_dataset_directory_binding.json",
    "subject00_generation_readiness_tests.json",
    "subject00_generation_readiness_final_summary.json",
]
DOC_OUTPUTS = [
    "AAAI27_SUBJECT00_DATA_PREPARATION_RECOVERY_20260725.md",
    "AAAI27_SUBJECT00_CONDITION_SLOT_BINDING_20260725.md",
    "AAAI27_SUBJECT00_GENERATION_BACKEND_SELECTION_20260725.md",
    "AAAI27_SUBJECT00_GENERATION_REQUEST_PLAN_20260725.md",
    "AAAI27_SUBJECT00_HUMAN_REVIEW_EXECUTION_PLAN_20260725.md",
]


class DuplicateKeyError(ValueError):
    pass


def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(key)
        result[key] = value
    return result


def load_json(path: Path, duplicate_scan: bool = True) -> dict[str, Any]:
    kwargs = {"object_pairs_hook": no_duplicates} if duplicate_scan else {}
    return json.loads(path.read_text(encoding="utf-8"), **kwargs)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_true(value: Any, message: str) -> None:
    if not value:
        raise AssertionError(message)


def run(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8")
    return completed.stdout.strip()


def git_rev(ref: str) -> str:
    return run(["git", "rev-parse", ref], ROOT)


def git_json(commit: str, relative: str) -> dict[str, Any]:
    return json.loads(run(["git", "show", f"{commit}:{relative}"], ROOT))


def ssh(command: str) -> str:
    return run(["ssh", "-o", "BatchMode=yes", "canondress-cloud", command])


def update_sealed(path: Path, field: str, value: Any) -> None:
    payload = load_json(path, duplicate_scan=False)
    payload.pop("content_sha256", None)
    payload[field] = value
    payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    loaded = {name: load_json(RISK / name) for name in JSON_OUTPUTS}
    recovery = loaded["subject00_data_preparation_recovery_audit.json"]
    identity = loaded["subject00_identity_source_registry.json"]
    inventory = loaded["subject00_condition_asset_inventory.json"]
    binding = loaded["subject00_condition_slot_binding.json"]
    split = loaded["subject00_controller_split_blueprint.json"]
    allowlist = loaded["subject00_garment_occlusion_allowlist.json"]
    taxonomy = loaded["subject00_garment_reject_taxonomy.json"]
    backend = loaded["subject00_generation_backend_selection.json"]
    prompts = loaded["subject00_generation_prompt_registry.json"]
    requests = loaded["subject00_generation_request_manifest.json"]
    reviews = loaded["subject00_human_review_execution_manifest.json"]
    directory = loaded["subject00_dataset_directory_binding.json"]
    summary = loaded["subject00_generation_readiness_final_summary.json"]
    handoff_path = ROOT / "project_control_handoff" / "subject00_data_preparation_generation_ready_handoff.json"
    handoff = load_json(handoff_path)

    checks: list[dict[str, Any]] = []

    def check(name: str, function: Callable[[], None]) -> None:
        try:
            function()
            checks.append({"name": name, "status": "PASS"})
        except Exception as error:  # noqa: BLE001 - report every gate failure
            checks.append({"name": name, "status": "FAIL", "detail": f"{type(error).__name__}: {error}"})

    check("strict_json_parse", lambda: [json.loads((RISK / name).read_text(encoding="utf-8")) for name in JSON_OUTPUTS])
    check("duplicate_key_scan", lambda: [load_json(RISK / name) for name in JSON_OUTPUTS])
    check(
        "required_local_path_existence",
        lambda: assert_true(
            all((RISK / name).is_file() for name in JSON_OUTPUTS)
            and all((ROOT / "docs" / "PAPER" / name).is_file() for name in DOC_OUTPUTS)
            and handoff_path.is_file(),
            "required output missing",
        ),
    )
    check(
        "sha256_format",
        lambda: assert_true(
            all(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in inventory["records"]),
            "invalid inventory SHA-256",
        ),
    )
    check("task_b_source_head", lambda: assert_true(git_rev("research/subject00-minimal-second-identity-dataset-contract-20260725") == SOURCE_HEAD, "Task B head changed"))
    check(
        "task_b_summary_handoff_git_consistency",
        lambda: assert_true(
            recovery["three_layer_consistency"]["status"] == "PASS"
            and git_json(SOURCE_HEAD, "paper_protocol/reviewer_risk/subject00_dataset_contract_final_summary.json")["classification"] == "SUBJECT00_CONDITION_RESOURCE_GAP"
            and git_json(SOURCE_HEAD, "project_control_handoff/subject00_minimal_second_identity_dataset_contract_handoff.json")["classification"] == "SUBJECT00_CONDITION_RESOURCE_GAP",
            "Task B source artifacts disagree",
        ),
    )
    check("slot_completeness_24", lambda: assert_true(len(binding["bindings"]) == 24 and binding["counts"]["source_bindings_complete"] == 24, "slot count is not 24"))
    check("reference_completeness_12", lambda: assert_true(sum("GARMENT_REFERENCE" in row["roles"] for row in binding["bindings"]) == 12, "reference count is not 12"))
    check("request_completeness_48", lambda: assert_true(len(requests["requests"]) == 48, "request count is not 48"))
    check("garment_count_3", lambda: assert_true(sorted({row["garment_id"] for row in binding["bindings"]}) == ["O01", "O03", "O04"], "garment set changed"))
    check(
        "slot_count_per_garment_8",
        lambda: assert_true(Counter(row["garment_id"] for row in binding["bindings"]) == Counter({"O01": 8, "O03": 8, "O04": 8}), "not eight slots per garment"),
    )
    check(
        "candidates_per_slot_2",
        lambda: assert_true(
            set(Counter((row["garment_id"], row["slot_id"]) for row in requests["requests"]).values()) == {2},
            "candidate count differs from two",
        ),
    )
    check("identity_source_uniqueness", lambda: assert_true(identity["identity_count"] == 1 and identity["identity_id"] == "subject00" and identity["view_count"] == 8, "identity source is not unique Subject00"))
    check(
        "strict_train_binding",
        lambda: assert_true(
            all(row["camera_id"] in TRAIN_CAMERAS and row["pose_frame_id"] == 0 and row["strict_split_role"] == "STRICT_TRAIN" for row in binding["bindings"]),
            "non-train source binding found",
        ),
    )
    check("heldout_leakage_zero", lambda: assert_true(not any(row["camera_id"] in HELDOUT_CAMERAS for row in binding["bindings"]), "heldout camera leakage"))
    check("buffer_pose_leakage_zero", lambda: assert_true(binding["strict_split"]["buffer_leakage_count"] == 0, "buffer pose leakage"))
    check(
        "controller_logical_overlap_zero",
        lambda: assert_true(
            all(not values for rotation in split["rotations"] for values in rotation["logical_overlap"].values())
            and split["logical_overlap_count"] == 0,
            "controller logical overlap found",
        ),
    )
    check(
        "secret_scan",
        lambda: assert_true(
            not re.search(
                r"(?i)(sk-[a-z0-9_-]{16,}|bearer\s+[a-z0-9._-]{16,}|api[_-]?key\s*[:=]\s*[\"'][^\"']{8,})",
                "\n".join((RISK / name).read_text(encoding="utf-8") for name in JSON_OUTPUTS),
            ),
            "secret-like value found",
        ),
    )
    check(
        "backend_evidence_audit",
        lambda: assert_true(
            backend["status"] == "BACKEND_SELECTION_REQUIRED"
            and backend["selection"] is None
            and len(backend["candidates"]) == 3
            and all(row["selection_result"] == "NOT_SELECTED" for row in backend["candidates"]),
            "backend selection is unsupported",
        ),
    )
    check(
        "authorization_false",
        lambda: assert_true(not any(row["authorized"] for row in requests["requests"]) and requests["authorized_count"] == 0, "authorized request found"),
    )
    check(
        "executed_false",
        lambda: assert_true(not any(row["executed"] for row in requests["requests"]) and requests["executed_count"] == 0, "executed request found"),
    )
    check("api_call_count_zero", lambda: assert_true(requests["image_generation_api_calls"] == 0 and backend["external_api_calls"] == 0 and summary["execution_counts"]["external_api"] == 0, "API count is nonzero"))
    check("materialized_output_count_zero", lambda: assert_true(binding["counts"]["materialized_teacher_targets"] == 0 and summary["counts"]["materialized_images"] == 0, "materialized image found"))
    check("formal_base_dependency_pending", lambda: assert_true(summary["formal_base_dependency"] == "PENDING" and not summary["formal_base_required_manifest_exists"], "Formal Base dependency overstated"))
    check("formal_base_root_absent", lambda: assert_true(ssh(f"if [ -e '{FORMAL_ROOT}' ]; then echo PRESENT; else echo ABSENT; fi") == "ABSENT", "Formal Base root unexpectedly present"))
    check("task_b_source_mutation_zero", lambda: assert_true(git_rev("research/subject00-minimal-second-identity-dataset-contract-20260725") == SOURCE_HEAD, "Task B branch mutated"))
    check("formal_protocol_mutation_zero", lambda: assert_true(git_rev("research/mmlphuman-subject00-formal-strict-split-protocol-20260723") == FORMAL_HEAD, "Formal protocol branch mutated"))
    check("storage_branch_mutation_zero", lambda: assert_true(git_rev("research/mmlphuman-subject00-storage-migration-adjudication-20260725") == STORAGE_HEAD, "Storage branch mutated"))

    def check_raw_hashes() -> None:
        selected = [row for row in inventory["records"] if row["category"] in {"identity_images_and_condition_rgb", "masks", "calibration_and_camera_poses", "smplx_parameters"}]
        paths = [row["absolute_path"] for row in selected]
        output = ssh("sha256sum " + " ".join(f"'{path}'" for path in paths))
        observed = {line.split()[1]: line.split()[0] for line in output.splitlines()}
        assert_true(all(observed.get(row["absolute_path"]) == row["sha256"] for row in selected), "raw selected asset SHA mismatch")

    check("raw_selected_asset_mutation_zero", check_raw_hashes)
    check(
        "paper_final_false",
        lambda: assert_true(
            all(not payload.get("paper_final", False) for payload in loaded.values()) and not handoff["paper_final"],
            "PAPER_FINAL is true",
        ),
    )
    check(
        "prompt_registry_hashes",
        lambda: assert_true(
            len(prompts["records"]) == 3
            and all(hashlib.sha256(row["positive_prompt"].encode()).hexdigest() == row["positive_prompt_sha256"] for row in prompts["records"])
            and all(hashlib.sha256(row["negative_constraints"].encode()).hexdigest() == row["negative_constraints_sha256"] for row in prompts["records"]),
            "prompt hash mismatch",
        ),
    )

    def check_request_hash() -> None:
        payload = [{key: value for key, value in row.items() if key != "request_manifest_sha256"} for row in requests["requests"]]
        expected = canonical_sha256(payload)
        assert_true(expected == requests["request_set_sha256"], "request-set hash mismatch")
        assert_true(all(row["request_manifest_sha256"] == expected for row in requests["requests"]), "per-request manifest hash mismatch")

    check("request_manifest_hash", check_request_hash)
    check("request_id_uniqueness_and_names", lambda: assert_true(len({row["request_id"] for row in requests["requests"]}) == 48 and all(re.fullmatch(r"subject00_O(?:01|03|04)_slot0[0-7]_cand0[01]", row["request_id"]) for row in requests["requests"]), "request IDs invalid"))
    check("review_manifest_empty_48", lambda: assert_true(reviews["candidate_count"] == 48 and len(reviews["candidates"]) == 48 and reviews["review_event_count"] == 0 and all(not row["initial_reviews"] and row["final_decision"] is None for row in reviews["candidates"]), "review results were invented"))
    check("human_review_policy", lambda: assert_true(reviews["decision_policy"]["minimum_independent_initial_reviewers"] == 2 and reviews["decision_policy"]["both_accept"] == "ACCEPT" and reviews["decision_policy"]["any_reject"] == "REJECT" and not reviews["decision_policy"]["vlm_final_tie_break_allowed"], "human review policy changed"))

    def check_cloud_directories() -> None:
        expected = [f"{DATASET_ROOT}/{row['name']}" for row in directory["directories"]]
        output = ssh("for p in " + " ".join(f"'{path}'" for path in expected) + "; do [ -d \"$p\" ] || exit 1; done; echo PASS")
        assert_true(output == "PASS" and len(expected) == 13, "cloud directory skeleton incomplete")

    check("cloud_directory_structure_13", check_cloud_directories)
    check("no_large_files_in_git", lambda: assert_true(not any(path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pth", ".pt", ".ckpt", ".npz"} for path in [*(RISK.rglob("*")), *((ROOT / "docs/PAPER").glob("AAAI27_SUBJECT00_*_20260725.*"))] if path.is_file()), "large/image asset copied into Git outputs"))
    check("orientation_tolerance", lambda: assert_true(all(row["orientation_error_degrees"] <= 22.5 for row in binding["bindings"]), "orientation tolerance exceeded"))
    check("occlusion_allowlist_strict", lambda: assert_true(not allowlist["allowlist_may_override_hard_reject"] and len(allowlist["never_allowed"]) >= 6, "occlusion allowlist is too weak"))
    check("reject_taxonomy_complete", lambda: assert_true(len(taxonomy["hard_rejects"]) >= 15 and all(row["decision"] == "REJECT" for row in taxonomy["hard_rejects"]), "reject taxonomy incomplete"))
    check("content_sha256_seals", lambda: [assert_true(canonical_sha256({key: value for key, value in payload.items() if key != "content_sha256"}) == payload["content_sha256"], f"seal mismatch {name}") for name, payload in loaded.items() if name != "subject00_generation_readiness_tests.json"])
    check("final_classification_allowed", lambda: assert_true(summary["classification"] == CLASSIFICATION and handoff["classification"] == CLASSIFICATION, "final classification mismatch"))
    check("next_task_exact", lambda: assert_true(summary["next_task"] == NEXT_TASK and handoff["next_task"] == NEXT_TASK and not summary["next_task_authorized"], "next task mismatch"))
    check("generation_not_ready_without_backend", lambda: assert_true(not summary["generation_readiness_gates"]["backend_selected"] and not summary["generation_may_start"], "generation readiness overstated"))
    check("generation_stop_rule", lambda: assert_true(summary["generation_stop_rule"] == "GENERATION_MAY_NOT_START_UNTIL_EXPLICIT_AUTHORIZATION_AND_FORMAL_BASE_ADJUDICATION", "generation stop rule missing"))

    failed = [row for row in checks if row["status"] == "FAIL"]
    test_status = f"PASS_{len(checks)}_READINESS_CHECKS" if not failed else f"FAIL_{len(failed)}_OF_{len(checks)}_READINESS_CHECKS"
    report = {
        "schema_version": "canondressgs.subject00.generation_readiness_tests.v1",
        "task_id": "AAAI27-SUBJECT00-DATA-PREPARATION-GENERATION-READY-001",
        "status": test_status,
        "checks": checks,
        "summary": {"total": len(checks), "passed": len(checks) - len(failed), "failed": len(failed)},
        "execution_boundary_verified": not failed,
        "paper_final": False,
    }
    report["content_sha256"] = canonical_sha256(report)
    report_path = RISK / "subject00_generation_readiness_tests.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    update_sealed(RISK / "subject00_generation_readiness_final_summary.json", "tests", test_status)
    update_sealed(handoff_path, "tests", test_status)

    print(json.dumps({"status": test_status, "total": len(checks), "failed": failed}, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
