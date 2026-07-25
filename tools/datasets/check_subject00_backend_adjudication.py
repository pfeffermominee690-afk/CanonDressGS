#!/usr/bin/env python3
"""Verify the evidence-only Subject00 backend adjudication contract."""

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
SOURCE_HEAD = "4e5e76a2a19ffdb94385660ec1d8ea4ce4c5d6b3"
WORKFLOW_HEAD = "e8a09525f745f373e38418c43040766d69a50d10"
TASK_B_HEAD = "9da04d923c7c6c03c6a6f7c732d8d9c58d59c88d"
STORAGE_HEAD = "34e91445ef45c001cebcd789a4a6a88cad7b9ad8"
FORMAL_STORAGE_HEAD = "f18eb0641afa2070ebeede3140f5c342c13944e9"
SOURCE_BRANCH = "research/subject00-data-preparation-generation-ready-20260725"
WORKFLOW_BRANCH = "research/multi-identity-generation-backend-review-workflow-20260724"
TASK_B_BRANCH = "research/subject00-minimal-second-identity-dataset-contract-20260725"
STORAGE_BRANCH = "research/mmlphuman-subject00-storage-migration-adjudication-20260725"
FORMAL_STORAGE_BRANCH = "research/mmlphuman-subject00-formal-storage-provisioned-run-20260725"
CLASSIFICATION = "SUBJECT00_GENERATION_BACKEND_SELECTION_REQUIRED"
NEXT_TASK = "USER_SELECT_SUBJECT00_GENERATION_PROVIDER_AND_COMPLETE_MANDATORY_BACKEND_EVIDENCE"
FORMAL_ROOT = "/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-FORMAL-STRICT-SPLIT-001"
DATASET_ROOT = "/root/autodl-tmp/canondressgs_work/datasets/subject00_three_garment"

JSON_OUTPUTS = [
    "subject00_backend_candidate_inventory.json",
    "subject00_backend_evidence_registry.json",
    "subject00_backend_mandatory_gate_matrix.json",
    "subject00_backend_weighted_comparison.json",
    "subject00_backend_selection_decision.json",
    "subject00_backend_request_schema.json",
    "subject00_backend_manifest_binding.json",
    "subject00_backend_cost_rate_limit_audit.json",
    "subject00_backend_minimal_canary_contract.json",
    "subject00_backend_secret_scan.json",
    "subject00_backend_adjudication_tests.json",
    "subject00_backend_adjudication_final_summary.json",
]
DOC_OUTPUTS = [
    "AAAI27_SUBJECT00_GENERATION_BACKEND_ADJUDICATION_20260725.md",
    "AAAI27_SUBJECT00_BACKEND_EVIDENCE_COMPARISON_20260725.md",
    "AAAI27_SUBJECT00_BACKEND_CANARY_PLAN_20260725.md",
]
HANDOFF_PATH = ROOT / "project_control_handoff" / "subject00_generation_backend_adjudication_handoff.json"
SCIENTIFIC_FIELDS = [
    "request_id", "garment_id", "slot_id", "semantic_pose_slot", "candidate_index",
    "identity_id", "identity_source_set_id", "camera_id", "pose_frame_id", "condition_id",
    "input_file_paths", "input_sha256", "source_condition_sha256", "prompt_id", "prompt_sha256",
    "negative_constraint_id", "negative_constraint_sha256", "output_count", "target_resolution", "output_format",
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_true(value: Any, message: str) -> None:
    if not value:
        raise AssertionError(message)


def run(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8")
    return completed.stdout.strip()


def git_rev(ref: str) -> str:
    return run(["git", "rev-parse", ref], ROOT)


def git_json(commit: str, path: str) -> dict[str, Any]:
    return json.loads(run(["git", "show", f"{commit}:{path}"], ROOT))


def ssh(command: str) -> str:
    return run(["ssh", "-o", "BatchMode=yes", "canondress-cloud", command])


def write_sealed(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    loaded = {name: load_json(RISK / name) for name in JSON_OUTPUTS}
    inventory = loaded["subject00_backend_candidate_inventory.json"]
    evidence = loaded["subject00_backend_evidence_registry.json"]
    matrix = loaded["subject00_backend_mandatory_gate_matrix.json"]
    weighted = loaded["subject00_backend_weighted_comparison.json"]
    decision = loaded["subject00_backend_selection_decision.json"]
    schema = loaded["subject00_backend_request_schema.json"]
    binding = loaded["subject00_backend_manifest_binding.json"]
    cost = loaded["subject00_backend_cost_rate_limit_audit.json"]
    canary = loaded["subject00_backend_minimal_canary_contract.json"]
    secret = loaded["subject00_backend_secret_scan.json"]
    summary = loaded["subject00_backend_adjudication_final_summary.json"]
    handoff = load_json(HANDOFF_PATH)
    source = load_json(RISK / "subject00_generation_request_manifest.json")
    source_rows = source["requests"]
    bound_rows = binding["bindings"]
    checks: list[dict[str, Any]] = []

    def check(name: str, function: Callable[[], None]) -> None:
        try:
            function()
            checks.append({"name": name, "status": "PASS"})
        except Exception as error:  # noqa: BLE001 - every failed gate must be reported
            checks.append({"name": name, "status": "FAIL", "detail": f"{type(error).__name__}: {error}"})

    check("strict_json_parse", lambda: [json.loads((RISK / name).read_text(encoding="utf-8")) for name in JSON_OUTPUTS] + [json.loads(HANDOFF_PATH.read_text(encoding="utf-8"))])
    check("duplicate_key_scan", lambda: [load_json(RISK / name) for name in JSON_OUTPUTS] + [load_json(HANDOFF_PATH)])
    check("required_output_paths", lambda: assert_true(all((RISK / name).is_file() for name in JSON_OUTPUTS) and all((ROOT / "docs" / "PAPER" / name).is_file() for name in DOC_OUTPUTS) and HANDOFF_PATH.is_file(), "required output missing"))
    check("data_prep_source_head", lambda: assert_true(git_rev(SOURCE_BRANCH) == SOURCE_HEAD, "Data Prep source head changed"))
    check("workflow_source_head", lambda: assert_true(git_rev(WORKFLOW_BRANCH) == WORKFLOW_HEAD, "workflow source head changed"))
    check("task_b_source_head", lambda: assert_true(git_rev(TASK_B_BRANCH) == TASK_B_HEAD, "Task B source head changed"))
    check("storage_source_head", lambda: assert_true(git_rev(STORAGE_BRANCH) == STORAGE_HEAD, "storage source head changed"))
    check("formal_source_head", lambda: assert_true(git_rev(FORMAL_STORAGE_BRANCH) == FORMAL_STORAGE_HEAD, "Formal source head changed"))
    check("source_classification", lambda: assert_true(git_json(SOURCE_HEAD, "paper_protocol/reviewer_risk/subject00_generation_readiness_final_summary.json")["classification"] == "SUBJECT00_DATA_PREPARATION_READY_PENDING_BACKEND", "source classification changed"))
    check("request_binding_48", lambda: assert_true(len(source_rows) == len(bound_rows) == binding["binding_count"] == 48, "binding count is not 48"))
    check("request_ids_unchanged", lambda: assert_true([row["request_id"] for row in source_rows] == [row["backend_request"]["request_id"] for row in bound_rows], "request IDs changed"))
    check("garment_slot_candidate_unchanged", lambda: assert_true(all(source_row["garment_id"] == bound_row["source_scientific_fields"]["garment_id"] and source_row["slot_id"] == bound_row["source_scientific_fields"]["slot_id"] and source_row["candidate_index"] == bound_row["source_scientific_fields"]["candidate_index"] for source_row, bound_row in zip(source_rows, bound_rows, strict=True)), "garment/slot/candidate changed"))
    check("garment_and_slot_counts", lambda: assert_true(Counter(row["garment_id"] for row in source_rows) == Counter({"O01": 16, "O03": 16, "O04": 16}) and len({(row["garment_id"], row["slot_id"]) for row in source_rows}) == 24, "frozen garment or slot budget changed"))
    check("input_shas_unchanged", lambda: assert_true(all(source_row["input_sha256"] == bound_row["backend_request"]["input_shas"] for source_row, bound_row in zip(source_rows, bound_rows, strict=True)), "input SHA changed"))
    check("scientific_fields_unchanged", lambda: assert_true(all({name: source_row[name] for name in SCIENTIFIC_FIELDS} == bound_row["source_scientific_fields"] for source_row, bound_row in zip(source_rows, bound_rows, strict=True)), "scientific request field changed"))
    check("scientific_field_seals", lambda: assert_true(all(canonical_sha256(row["source_scientific_fields"]) == row["source_scientific_fields_sha256"] for row in bound_rows), "scientific binding seal mismatch"))
    check("authorization_false", lambda: assert_true(binding["authorized_count"] == 0 and not any(row["backend_request"]["authorization"] for row in bound_rows) and not decision["generation_authorized"], "request was authorized"))
    check("executed_false", lambda: assert_true(binding["executed_count"] == 0 and not any(row["backend_request"]["executed"] for row in bound_rows) and not decision["generation_started"], "request was executed"))
    check("raw_responses_zero", lambda: assert_true(binding["raw_response_count"] == summary["raw_responses"] == 0 and all(row["response_count"] == 0 for row in bound_rows), "raw response count is nonzero"))
    check("generated_images_zero", lambda: assert_true(binding["materialized_image_count"] == summary["materialized_images"] == 0 and all(row["materialized_image_count"] == 0 for row in bound_rows), "generated image count is nonzero"))
    check("api_calls_zero", lambda: assert_true(decision["api_calls"] == summary["api_calls"] == summary["connectivity_calls"] == summary["paid_calls"] == handoff["api_calls"] == 0, "API call count is nonzero"))
    check("secret_exposure_zero", lambda: assert_true(secret["status"] == "PASS_NO_SECRET_EXPOSURE" and not secret["credential_value_read"] and secret["secret_like_matches"] == summary["credential_values_read"] == 0, "secret exposure recorded"))

    def check_secret_patterns() -> None:
        text = "\n".join((RISK / name).read_text(encoding="utf-8") for name in JSON_OUTPUTS)
        matches = re.search(r"(?i)(sk-[a-z0-9_-]{16,}|bearer\s+[a-z0-9._-]{16,}|api[_-]?key\s*[:=]\s*[\"'][^\"']{8,})", text)
        assert_true(matches is None, "secret-like value found")

    check("generated_artifact_secret_scan", check_secret_patterns)
    check("candidate_inventory_complete", lambda: assert_true(inventory["candidate_count"] == len(inventory["candidates"]) == 3 and {row["backend_id"] for row in inventory["candidates"]} == {"BACKEND-A", "BACKEND-B", "BACKEND-C"}, "candidate inventory incomplete"))
    check("mandatory_gate_matrix_complete", lambda: assert_true(matrix["gate_count"] == 15 and len(matrix["candidates"]) == 3 and all(len(row["gates"]) == 15 and {gate["gate_id"] for gate in row["gates"]} == set(matrix["gate_ids"]) for row in matrix["candidates"]), "mandatory gate matrix incomplete"))
    check("no_backend_gate_eligible", lambda: assert_true(matrix["eligible_backend_count"] == 0 and all(not row["all_mandatory_gates_pass"] and not row["direct_selection_eligible"] and not row["pending_canary_selection_eligible"] for row in matrix["candidates"]), "unsupported backend eligibility"))
    check("weighted_comparison_auxiliary", lambda: assert_true(len(weighted["dimensions"]) == 19 and len(weighted["candidates"]) == 3 and all(row["mandatory_gate_override"] == "INELIGIBLE" and row["total"] == sum(row["scores"].values()) for row in weighted["candidates"]), "weighted comparison invalid"))
    check("provider_model_null_when_unselected", lambda: assert_true(decision["primary"] is None and decision["fallback"] is None and decision["provider"] is None and decision["model"] is None and binding["provider_model_revision_non_null_count"] == 0 and all(row["backend_request"]["backend"] is None and row["backend_request"]["provider"] is None and row["backend_request"]["model"] is None for row in bound_rows), "unselected provider/model is non-null"))
    check("request_schema_strict", lambda: assert_true(schema["type"] == "object" and schema["additionalProperties"] is False and len(schema["required"]) == 24 and set(schema["required"]) == set(schema["properties"]), "backend request schema is not strict"))
    check("binding_schema_fields_complete", lambda: assert_true(all(set(row["backend_request"]) == set(schema["required"]) for row in bound_rows), "binding does not match strict request fields"))
    check("target_resolution_frozen", lambda: assert_true(all(row["backend_request"]["target_resolution"] == {"height": 1536, "orientation": "portrait", "width": 1024} for row in bound_rows), "target resolution changed"))
    check("canary_uses_existing_budget", lambda: assert_true(canary["request_count"] == 2 and canary["request_ids"] == ["subject00_O01_slot00_cand00", "subject00_O03_slot03_cand00"] and set(canary["request_ids"]).issubset({row["request_id"] for row in source_rows}), "canary request not in frozen 48"))
    check("canary_not_authorized", lambda: assert_true(not canary["authorization"] and not canary["started"] and canary["api_calls"] == canary["generated_images"] == 0, "canary was started"))
    check("formal_base_pending", lambda: assert_true(summary["formal_base"]["dependency"] == "PENDING" and not summary["formal_base"]["sealed_manifest_exists"] and summary["formal_base"]["execution_head"] is None, "Formal Base overstated"))
    check("formal_root_absent", lambda: assert_true(ssh(f"if [ -e '{FORMAL_ROOT}' ]; then echo PRESENT; else echo ABSENT; fi") == "ABSENT", "Formal root unexpectedly present"))
    check("storage_blocked", lambda: assert_true(summary["storage"]["status"] == "BLOCKED_CAPACITY" and summary["storage"]["live_observed_free_bytes"] < summary["storage"]["required_free_bytes"] and not summary["storage"]["plan_executed"], "storage state overstated"))
    check("cloud_dataset_still_empty", lambda: assert_true(ssh(f"find '{DATASET_ROOT}' -type f | wc -l") == "0", "dataset root contains materialized files"))

    def check_raw_hashes() -> None:
        expected: dict[str, str] = {}
        for row in source_rows:
            for key, path in row["input_file_paths"].items():
                sha = row["input_sha256"].get(key)
                if path and sha:
                    expected[path] = sha
        output = ssh("sha256sum " + " ".join(f"'{path}'" for path in sorted(expected)))
        observed = {line.split()[1]: line.split()[0] for line in output.splitlines()}
        assert_true(all(observed.get(path) == sha for path, sha in expected.items()), "raw source asset SHA mismatch")

    check("raw_dataset_mutation_zero", check_raw_hashes)

    def check_evidence_hashes() -> None:
        rows = [row for row in evidence["records"] if row["sha256"] is not None and not row["path"].startswith("git:")]
        for row in rows:
            path = Path(row["path"])
            assert_true(path.is_file() and path.stat().st_size == row["bytes"] and file_sha256(path) == row["sha256"], f"evidence changed: {path}")

    check("local_evidence_hashes", check_evidence_hashes)
    check("historical_success_classes_separated", lambda: assert_true(evidence["distinctions"]["CONNECTIVITY_ONLY"] and evidence["distinctions"]["IMAGE_EDIT_SUCCESS"] and evidence["distinctions"]["FORMAL_DATASET_GENERATION_SUCCESS"] and not evidence["distinctions"]["SUBJECT02_FORMAL_TARGET_SUCCESS"], "historical evidence classes collapsed"))
    check("price_rate_not_guessed", lambda: assert_true(not cost["price_guessed"] and not cost["rate_limit_guessed"] and all(row["per_call_cost"] == "PRICE_NOT_VERIFIED" and row["rpm_tpm_or_image_limit"] == "RATE_LIMIT_NOT_VERIFIED" for row in cost["candidates"]), "price or rate was guessed"))
    check("content_sha256_seals", lambda: [assert_true(canonical_sha256({key: value for key, value in payload.items() if key != "content_sha256"}) == payload["content_sha256"], f"seal mismatch: {name}") for name, payload in loaded.items() if name != "subject00_backend_adjudication_tests.json"] + [assert_true(canonical_sha256({key: value for key, value in handoff.items() if key != "content_sha256"}) == handoff["content_sha256"], "handoff seal mismatch")])
    check("paper_final_false", lambda: assert_true(all(not payload.get("paper_final", False) for payload in loaded.values()) and not handoff["paper_final"], "PAPER_FINAL is true"))
    check("final_classification", lambda: assert_true(decision["classification"] == summary["classification"] == handoff["classification"] == CLASSIFICATION, "classification mismatch"))
    check("next_task_exact", lambda: assert_true(decision["next_task"] == summary["next_task"] == handoff["next_task"] == NEXT_TASK and not decision["next_task_authorized"] and not summary["next_task_authorized"] and not handoff["next_task_authorized"], "next task mismatch"))

    def check_no_binary_outputs() -> None:
        changed = run(["git", "diff", "--name-only", SOURCE_HEAD, "--"], ROOT).splitlines()
        forbidden = {".png", ".jpg", ".jpeg", ".webp", ".npz", ".npy", ".pt", ".pth", ".ckpt"}
        assert_true(not any(Path(path).suffix.lower() in forbidden for path in changed), "binary/generated asset added")

    check("no_generated_or_binary_assets_added", check_no_binary_outputs)

    failed = [row for row in checks if row["status"] == "FAIL"]
    status = f"PASS_{len(checks)}_BACKEND_ADJUDICATION_CHECKS" if not failed else f"FAIL_{len(failed)}_OF_{len(checks)}"
    test_payload = {
        "schema_version": "canondressgs.subject00.backend_adjudication_tests.v1",
        "task_id": "AAAI27-SUBJECT00-GENERATION-BACKEND-ADJUDICATION-001",
        "status": status,
        "total": len(checks),
        "passed": len(checks) - len(failed),
        "failed": failed,
        "checks": checks,
        "paper_final": False,
    }
    write_sealed(RISK / "subject00_backend_adjudication_tests.json", test_payload)

    for path in [RISK / "subject00_backend_adjudication_final_summary.json", HANDOFF_PATH]:
        payload = load_json(path, duplicate_scan=False)
        payload["tests"] = status
        write_sealed(path, payload)

    print(json.dumps({"status": status, "total": len(checks), "failed": failed}, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
