from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_ROOT = REPO_ROOT / "paper_protocol" / "subject00_formal_base"

SUMMARY = PREFLIGHT_ROOT / "subject00_formal_base_101245_preflight_summary_20260726.json"
MANIFEST = (
    PREFLIGHT_ROOT
    / "subject00_formal_base_101245_execution_manifest_draft_20260726.json"
)
TESTS = PREFLIGHT_ROOT / "subject00_formal_base_101245_preflight_tests_20260726.json"
HANDOFF = PREFLIGHT_ROOT / "subject00_formal_base_101245_handoff_20260726.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    summary = load_json(SUMMARY)
    manifest = load_json(MANIFEST)
    tests = load_json(TESTS)
    handoff = load_json(HANDOFF)

    fields = summary["final_response_fields"]
    require(fields["TASK_ID"] == "AAAI27-SUBJECT00-FORMAL-BASE-EXECUTION-PREFLIGHT-001", "task id changed")
    require(fields["SOURCE_HEAD"] == "d541e46a7963502d1e6613cbdfa477fdb4e82d6c", "source head changed")
    require(fields["TOTAL_OPTIMIZER_STEPS"] == 101245, "optimizer step count changed")
    require(fields["TRAINING_AUTHORIZED"] is False, "training authorization must remain false")
    require(fields["OPTIMIZER_STEPS"] == 0, "preflight must not perform optimizer steps")
    require(fields["OUTPUT_ROOT_CREATED"] is False, "preflight must not create output root")
    require(fields["DATA_MUTATIONS"] == 0, "data mutation counter changed")
    require(fields["CHECKPOINT_MUTATIONS"] == 0, "checkpoint mutation counter changed")
    require(fields["PAPER_MODIFICATIONS"] == 0, "paper mutation counter changed")
    require(fields["INIT_CHECKPOINT_STEP"] == 0, "initial checkpoint step changed")
    require(
        fields["INIT_CHECKPOINT_SHA256"]
        == "29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a",
        "initial checkpoint SHA changed",
    )
    require(
        fields["OUTPUT_ROOT"]
        == "/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-FORMAL-BASE-101245-001",
        "formal output root changed",
    )
    require(
        fields["FINAL_CLASSIFICATION"]
        == "SUBJECT00_FORMAL_BASE_EXECUTION_CONTRACT_READY_PENDING_USER_AUTHORIZATION",
        "final classification changed",
    )
    require(
        fields["NEXT_TASK"]
        == "USER_AUTHORIZE_SUBJECT00_FORMAL_BASE_101245_STEP_EXECUTION",
        "next task changed",
    )

    require(manifest["training_authorized"] is False, "manifest authorizes training")
    require(manifest["optimizer_steps_executed_by_preflight"] == 0, "manifest optimizer counter changed")
    require(manifest["output_root_created_by_preflight"] is False, "manifest output root counter changed")
    require(manifest["initialization"]["checkpoint_step"] == 0, "manifest init step changed")
    require(manifest["formal_training"]["total_optimizer_steps"] == 101245, "manifest step count changed")
    require(manifest["storage"]["formal_storage_gate_bytes"] == 32212254720, "storage gate changed")
    require(manifest["storage"]["live_free_bytes"] >= 32212254720, "live free space no longer passes gate")

    test_rows = tests["tests"]
    require(len(test_rows) >= 44, "required machine test coverage is incomplete")
    failing = [row for row in test_rows if not str(row["status"]).startswith("PASS")]
    require(not failing, f"preflight test failures: {failing}")
    require(tests["optimizer_steps_executed_by_tests"] == 0, "tests performed optimizer steps")
    require(tests["output_root_created_by_tests"] is False, "tests created output root")

    require(handoff["training_authorized"] is False, "handoff authorizes training")
    require(handoff["next_task"] == fields["NEXT_TASK"], "handoff next task diverges")

    print("SUBJECT00_FORMAL_BASE_PREFLIGHT_CONTRACT_BUNDLE_PASS")


if __name__ == "__main__":
    main()
